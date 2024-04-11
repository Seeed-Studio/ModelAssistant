# copyright Copyright (c) Seeed Technology Co.,Ltd.
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmdet.models.utils import multi_apply, unpack_gt_instances
from mmengine.model import BaseModule, constant_init, normal_init
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F

from sscma.models.base import is_norm
from sscma.registry import LOSSES

from ..base.general import CBR


@MODELS.register_module()
class FomoHead(BaseModule):
    """The difference between the Fomo model head and the target detection
    model head is that the output of this model only contains the probability
    of all categories, and does not contain the value of xyhw."""

    def __init__(
        self,
        input_channels: Union[Sequence[int], int],
        middle_channel: int = 48,
        num_classes: int = 20,
        act_cfg: str = 'ReLU6',
        loss_weight: Optional[Sequence[int]] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        loss_cls: Optional[dict] = dict(type='BCEWithLogitsLoss', reduction='mean'),
        loss_bg: Optional[dict] = dict(type='BCEWithLogitsLoss', reduction='mean'),
        init_cfg: Optional[dict] = dict(type='Normal', std=0.01),
    ) -> None:
        super(FomoHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.input_channels = input_channels if isinstance(input_channels, Sequence) else [input_channels]

        self.middle_channels = middle_channel[0] if isinstance(middle_channel, Sequence) else middle_channel
        self.act_cfg = act_cfg

        if loss_weight:
            for idx, w in enumerate(loss_weight):
                self.weight_cls[idx + 1] = w

        self.loss_bg = LOSSES.build(loss_bg)
        self.loss_cls = LOSSES.build(loss_cls)

        # Offset of the ground truth box
        self.posit_offset = torch.tensor(
            [
                [0, -1, 0],
                [0, -1, -1],
                [0, 0, -1],
                [0, 1, 0],
                [0, 1, 1],
                [0, 0, 1],
                [0, 1, -1],
                [0, -1, 1],
                [0, 0, 0],
            ],
            dtype=torch.long,
        )
        self._init_layers()

    def _init_layers(self):
        self.convs_bridge = nn.ModuleList()
        self.convs_pred = nn.ModuleList()
        for i in range(len(self.input_channels)):
            self.convs_bridge.append(
                CBR(
                    self.input_channels[i],
                    self.middle_channels,
                    3,
                    1,
                    padding=1,
                    act=self.act_cfg,
                )
            )
            self.convs_pred.append(nn.Conv2d(self.middle_channels, self.num_attrib, 1, padding=0))

    def forward(self, x: Tuple[torch.Tensor, ...]):
        """
        Forward features from the upstream network.
        Args:
            x(tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            (tuple): Output the corresponding number of predicted feature
                maps according to the output of the upstream network

        """
        assert len(x) == len(self.input_channels)

        result = []
        for i, feat in enumerate(x):
            feat = self.convs_bridge[i](feat)
            pred_map = self.convs_pred[i](feat)
            result.append(pred_map)

        return tuple(result)

    def loss(self, inputs: Tuple[torch.Tensor, ...], data_samples):
        pred = self.forward(inputs)
        gt = unpack_gt_instances(data_samples)
        (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = gt

        loss = self.loss_by_feat(pred, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore)

        return loss

    def predict(self, features, batch_data_samples, rescale=False):
        preds = self.forward(features)
        preds = tuple([F.softmax(pred, dim=1) for pred in preds])

        batch_data_samples[0].metainfo['img_shape']
        # batch_gt_instances = [data_samples.gt_instances for data_samples in batch_data_samples]
        batch_gt_instances = [data_samples.fomo_mask for data_samples in batch_data_samples]

        return [
            InstanceData(
                pred=preds,
                labels=tuple(
                    [
                        torch.concat(
                            [torch.from_numpy(batch_gt_instances[i][0]) for i in range(len(batch_gt_instances))]
                        ).to(preds[0].device)
                    ]
                ),
            )
        ]

    def loss_by_feat(self, preds, batch_gt_instances, batch_img_metas, batch_gt_instances_ignore) -> dict:
        device = preds[0].device
        batch_img_metas[0]['img_shape']  # batch_input_shape
        # Get the ground truth box that fits the fomo model
        # target = [self.build_target(pred.shape[2:], input_shape, batch_gt_instances, device) for pred in preds]
        target = [
            torch.concat(
                [torch.from_numpy(batch_img_metas[i]['fomo_mask'][0]) for i in range(len(batch_img_metas))]
            ).to(device)
        ]
        loss, cls_loss, bg_loss, P, R, F1 = multi_apply(self.lossFunction, preds, target)

        return dict(loss=loss, fgnd=cls_loss, bgnd=bg_loss, P=P, R=R, F1=F1)

    def lossFunction(self, pred_maps: torch.Tensor, data: torch.Tensor):
        """Calculate the loss of the model
        Args:
            preds(torch.Tensor): Model Predicted Output
            target(torch.Tensor): The target feature map constructed according to the
                size of the feature map output by the model

        Returns:
            (dict): The model loss value in the training phase and the evaluation index
                of the model
        """
        preds = pred_maps.permute(0, 2, 3, 1)
        B, H, W, C = preds.shape
        # pos_weights
        if not hasattr(self, 'weight_mask'):
            weight = torch.zeros(self.num_attrib, device=preds.device)
            weight[0] = 1
            self.weight_mask = torch.tile(weight, (H, W, 1))

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
        P, R, F1 = self.get_pricsion_recall_f1(preds, data)
        return (
            loss,
            cls_loss,
            bg_loss,
            torch.Tensor([P]),
            torch.Tensor([R]),
            torch.Tensor([F1]),
        )

    def get_pricsion_recall_f1(self, preds: torch.Tensor, target: torch.Tensor):
        """Calculate the evaluation index of model prediction according to the
        prediction result of the model and the target feature map.

        Args:
            preds(torch.Tensor): Model Predicted Output
            target(torch.Tensor): The target feature map constructed according to the
                size of the feature map output by the model

        Returns:
            P: Precision
            R: Recall
            F1: F1
        """
        with torch.no_grad():
            target = target.to(device=torch.device('cpu'), non_blocking=True).numpy()
            preds = torch.softmax(preds, dim=-1).to(device=torch.device('cpu'), non_blocking=True).numpy()

        B, H, W, C = preds.shape
        # Get the category id of each box
        target_max = np.argmax(target, axis=-1)
        preds_max = np.argmax(preds, axis=-1)
        # Get the index of the forecast for the non-background
        target_condition = np.where(target_max > 0)
        preds_condition = np.where(preds_max > 0)
        # splice index
        target_index = np.stack(target_condition, axis=1)
        preds_index = np.stack(preds_condition, axis=1)
        # Traversal compares predicted and ground truth boxes
        for ti in target_index:
            for po in self.posit_offset:
                site = np.sum([ti, po], axis=0)
                # Avoid index out of bounds
                if np.any(site < 0) or np.any(site >= H):
                    continue
                # The prediction is considered to be correct if it is near the ground truth box
                if site in preds_index:
                    sc, tc = tuple(np.split(site, 3)), tuple(np.split(ti, 3))
                    if preds_max[sc] == target_max[tc]:
                        preds_max[sc] = target_max[tc]
                        target_max[sc] = target_max[tc]
        # Calculate the confusion matrix
        confusion = confusion_matrix(target_max.flatten(), preds_max.flatten(), labels=range(self.num_attrib))
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

    @property
    def num_attrib(self):
        """The number of classifications the model needs to classify (including background)
        Return:
            (int): Add one to the number of categories
        """
        return self.num_classes + 1

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01)
            if is_norm(m):
                constant_init(m, 1)
