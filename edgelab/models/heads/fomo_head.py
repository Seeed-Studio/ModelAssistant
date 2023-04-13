from typing import Optional, List

import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import normal_init, constant_init, is_norm

from ..base.general import CBR


@HEADS.register_module()
class Fomo_Head(BaseModule):

    def __init__(
        self,
        input_channels: int,
        middle_channels: List[int] = [96, 32],
        num_classes: int = 20,
        act_cfg: str = 'ReLU6',
        cls_weight: int = 1,
        loss_weight: List[int] = None,
        train_cfg: dict = None,
        test_cfg: dict = None,
        loss_cls: dict = dict(type='BCEWithLogitsLoss', reduction='mean'),
        loss_bg: dict = dict(type='BCEWithLogitsLoss', reduction='mean'),
        init_cfg: Optional[dict] = dict(type='Normal', std=0.01)
    ) -> None:
        super(Fomo_Head, self).__init__(init_cfg)
        self.num_classes = num_classes

        if loss_weight:
            for idx, w in enumerate(loss_weight):
                self.weight_cls[idx + 1] = w
        self.loss_cls = nn.BCEWithLogitsLoss(reduction='none',
                                             pos_weight=torch.Tensor(
                                                 [cls_weight]))
        self.loss_bg = nn.BCEWithLogitsLoss(reduction='none')

        # Offset of the ground truth box
        self.posit_offset = torch.tensor(
            [[0, -1, 0], [0, -1, -1], [0, 0, -1], [0, 1, 0], [0, 1, 1],
             [0, 0, 1], [0, 1, -1], [0, -1, 1], [0, 0, 0]],
            dtype=torch.long)

        self.conv1 = CBR(input_channels,
                         middle_channels[0],
                         1,
                         1,
                         padding=0,
                         act=act_cfg)
        self.conv2 = CBR(middle_channels[0],
                         middle_channels[1],
                         1,
                         1,
                         padding=0,
                         act='ReLU')

        self.conv3 = nn.Conv2d(middle_channels[1],
                               num_classes + 1,
                               1,
                               1,
                               padding=0)

    def forward(self, x):
        if isinstance(x, (list,tuple)) and len(x):
            x = x[-1]
        x = self.conv1(x)
        x = self.conv2(x)
        result = self.conv3(x)

        return result

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        results = self(x)
        loss = self.loss(results,
                         gt_bboxes=gt_bboxes,
                         gt_labels=gt_labels,
                         gt_bbox_ignore=gt_bboxes_ignore,
                         img_metas=img_metas)
        return loss

    def loss(self, pred_maps: torch.Tensor, target):
        """ Calculate the loss of the model """
        preds = pred_maps.permute(0, 2, 3, 1)
        B, H, W, C = preds.shape
        # pos_weights
        weight = torch.zeros(self.num_attrib, device=preds.device)
        weight[0] = 1
        self.weight_mask = torch.tile(weight, (H, W, 1))
        # Get the ground truth box that fits the fomo model
        data = self.build_target(preds, target)
        # background loss
        bg_loss = self.loss_bg(
            preds,
            data,
        )
        bg_loss *= self.weight_mask
        # no background loss
        cls_loss = self.loss_cls(
            preds,
            data,
        )
        cls_loss *= 1.0 - self.weight_mask
        # avg loss
        loss = torch.mean(cls_loss + bg_loss)
        # get p,r,f1
        P, R, F1 = self.get_pricsion_revall_f1(preds, data)
        return dict(loss=loss,
                    fgnd=cls_loss,
                    bgnd=bg_loss,
                    P=torch.Tensor([P]),
                    R=torch.Tensor([R]),
                    F1=torch.Tensor([F1]))

    def get_pricsion_revall_f1(self, preds, target):
        preds = torch.softmax(preds, dim=-1)
        # Get the category id of each box
        target_max = torch.argmax(target, dim=-1)
        preds_max = torch.argmax(preds, dim=-1)
        # Get the index of the forecast for the non-background
        target_condition = torch.where(target_max > 0)
        preds_condition = torch.where(preds_max > 0)
        # splice index
        target_index = torch.stack(target_condition, dim=1)
        preds_index = torch.stack(preds_condition, dim=1)

        self.posit_offset = self.posit_offset.to(target.device)
        # Traversal compares predicted and ground truth boxes
        for ti in target_index:
            for po in self.posit_offset:
                site = ti + po
                # Avoid index out ofAvoid index out of bounds
                if torch.any(site < 0) or torch.any(site > 11):
                    continue
                # The prediction is considered to be correct if it is near the ground truth box
                if site in preds_index and preds_max[site.chunk(
                        3)] == target_max[ti.chunk(3)]:
                    preds_max[site.chunk(3)] = target_max[ti.chunk(3)]
                    target_max[site.chunk(3)] = target_max[ti.chunk(3)]
        # Calculate the confusion matrix
        confusion = confusion_matrix(target_max.flatten().cpu().numpy(),
                                     preds_max.flatten().cpu().numpy(),
                                     labels=range(self.num_attrib))
        # Calculate the value of P、R、F1 based on the confusion matrix
        tn = confusion[0, 0]
        tp = np.diagonal(confusion).sum() - tn
        fn = np.tril(confusion, k=-1).sum()
        fp = np.triu(confusion, k=1).sum()
        # Denominator cannot be zero
        if tp + fp == 0 or tp + fn == 0:
            return 0.0, 0.0, 0.0
        # calculate
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * (p * r) / (p + r) if p + r != 0 else 0

        return p, r, f1

    def post_handle(self, preds, target):
        preds = preds.permute(0, 2, 3, 1)
        B, H, W, C = preds.shape
        assert self.num_attrib == C

        mask = torch.softmax(preds, dim=-1)
        values, indices = torch.max(mask, dim=-1)
        values_mask = np.argwhere(values.cpu().numpy() < 0.25)
        res = torch.argmax(mask, dim=-1)

        for i in values_mask:
            b, h, w = int(i[0].item()), int(i[1].item()), int(i[2].item())
            res[b, h, w] = 0

        return preds, self.build_target(preds, target)

    def build_target(self, preds, targets):
        B, H, W, C = preds.shape
        target_data = torch.zeros(size=(B, H, W, C), device=preds.device)
        target_data[..., 0] = 1

        for i in targets:
            h, w = int(i[3].item() * H), int(i[2].item() * W)
            target_data[int(i[0]), h, w, 0] = 0  # background
            target_data[int(i[0]), h, w, int(i[1])] = 1  #label

        return target_data

    def merge_gt(self, gt_bboxes, gt_labels, img_metas):
        target = []
        max_size = max(img_metas[0]['img_shape'])
        for idx, (labels, bboxes) in enumerate(zip(gt_labels, gt_bboxes)):
            bboxes = bboxes / max_size

            bb = torch.zeros_like(bboxes, device=bboxes.device)
            bb[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / 2
            bb[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / 2
            bb[..., 2] = bboxes[..., 2] - bboxes[..., 0]
            bb[..., 3] = bboxes[..., 3] - bboxes[..., 1]

            num = bb.shape[0]
            labels = labels.reshape((num, 1))
            z0 = torch.zeros((num, 1), device=bboxes.device) + idx
            bb = torch.concat((labels, bb), 1)
            gt = torch.concat((z0, bb), axis=1)
            target.append(gt)

        target = torch.concat(target, 0)

        return target

    @property
    def num_attrib(self):
        """ The number of classifications the model needs to classify (including background) """
        return self.num_classes + 1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
