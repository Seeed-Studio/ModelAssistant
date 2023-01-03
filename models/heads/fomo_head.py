from typing import Optional

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS, build_loss
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import normal_init, constant_init, is_norm

from models.base.general import CBR


@HEADS.register_module()
class Fomo_Head(BaseModule):

    def __init__(
        self,
        input_channels: int,
        middle_channels: int = 32,
        num_classes: int = 20,
        train_cfg: dict = None,
        test_cfg: dict = None,
        loss_cls: dict = dict(type='BCEWithLogitsLoss', reduction='mean'),
        loss_cls_no: dict = dict(type='BCEWithLogitsLoss', reduction='mean'),
        init_cfg: Optional[dict] = dict(type='Normal', std=0.01)
    ) -> None:
        super(Fomo_Head, self).__init__(init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_cls_no = build_loss(loss_cls_no)

        self.num_classes = num_classes
        self.conv1 = CBR(input_channels, middle_channels, 1, 1, padding=0)

        self.conv2 = nn.Conv2d(middle_channels, num_classes + 1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        result = self.conv2(x)
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

    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bbox_ignore=None):

        target = self.merge_gt(gt_bboxes=gt_bboxes,
                               gt_labels=gt_labels,
                               img_metas=img_metas)
        preds = pred_maps.permute(0, 2, 3, 1)

        weight = torch.zeros(21, device=preds.device)
        weight[0] = 1

        data = self.build_target(preds, target)

        cls_loss = self.loss_cls(preds, data)

        cls_no_loss = self.loss_cls_no(preds, data)

        loss = cls_loss * 100 + cls_no_loss
        loss_dict = {}
        loss_dict['loss'] = loss
        loss_dict['cls_no_loss'] = cls_no_loss
        loss_dict['cls_loss'] = cls_loss

        return dict(loss=loss, cls_loss=cls_loss, cls_no_loss=cls_no_loss)

    def post_handle(self, preds):
        preds = preds.permute(0, 2, 3, 1)
        B, H, W, C = preds.shape
        assert self.num_attrib == C

        mask = torch.softmax(preds, dim=-1)
        values, indices = torch.max(mask, dim=-1)
        values_mask = torch.argwhere(values < 0.5)
        res = torch.argmax(mask, dim=-1)

        for i in values_mask:
            b, h, w = int(i[0].cpu().item()), int(i[1].cpu().item()), int(
                i[2].cpu().item())
            res[b, h, w] = 0
        return res

    def build_target(self, preds, targets):
        B, H, W, C = preds.shape
        target_data = torch.zeros(size=(B, H, W, 21), device=preds.device)
        target_data[..., 0] = 1
        for i in targets:
            h, w = round(i[3].item() * (H - 1)), round(i[2].item() * (W - 1))
            target_data[int(i[0]), h, w, 0] = 0  #confnes
            target_data[int(i[0]), h, w, int(i[1]) + 1] = 1  #label

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
        return self.num_classes + 1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
