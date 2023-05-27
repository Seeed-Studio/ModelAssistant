from typing import Union, Sequence, Tuple, List
import math

import torch
import torch.nn as nn
from torch import Tensor
from mmengine.model import BaseModel
from mmdet.structures import SampleList
from mmdet.models.utils import multi_apply, unpack_gt_instances
from mmdet.utils import InstanceList, OptMultiConfig
from mmyolo.models.utils import make_divisible
from mmyolo.models.dense_heads.yolov5_head import YOLOv5Head

from edgelab.registry import MODELS


@MODELS.register_module()
class DetHead(BaseModel):

    def __init__(self,
                 num_classes: int,
                 in_channels: Union[int, Sequence],
                 widen_factor: float = 1.0,
                 num_base_priors: int = 3,
                 featmap_strides: Sequence[int] = (8, 16, 32),
                 anchors=[[(10, 13), (16, 30), (33, 23)],
                          [(30, 61), (62, 45), (59, 119)],
                          [(116, 90), (156, 198), (373, 326)]],
                 init_cfg: OptMultiConfig = None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.widen_factor = widen_factor

        self.na = len(anchors[0])

        self.anchors = torch.as_tensor(anchors)
        self.featmap_strides = featmap_strides
        self.num_out_attrib = 5 + self.num_classes
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors

        if isinstance(in_channels, int):
            self.in_channels = [make_divisible(in_channels, widen_factor)
                                ] * self.num_levels
        else:
            self.in_channels = [
                make_divisible(i, widen_factor) for i in in_channels
            ]

        self._init_layers()

    def _init_layers(self):
        """initialize conv layers in YOLOv5 head."""
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_base_priors * self.num_out_attrib,
                                  1)

            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize the bias of YOLOv5 head."""
        super().init_weights()
        for mi, s in zip(self.convs_pred, self.featmap_strides):  # from
            b = mi.bias.data.view(self.num_base_priors, -1)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s)**2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.999999))

            mi.bias.data = b.view(-1)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions, and objectnesses.
        """
        assert len(x) == self.num_levels

        return multi_apply(self.forward_split, x, self.convs_pred)

    def _forward(self, x) -> List[Tensor]:
        assert len(x) == self.num_levels
        res = []
        for feat, conv in zip(x, self.convs_pred):
            res.append(self.forward_single(feat, conv))
        return self.process(res)

    def forward_split(self, x: Tensor, convs: nn.Module):
        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, self.num_base_priors, self.num_out_attrib,
                                 ny, nx)
        cls_score = pred_map[:, :, 5:, ...].reshape(bs, -1, ny, nx)
        bbox_pred = pred_map[:, :, :4, ...].reshape(bs, -1, ny, nx)
        objectness = pred_map[:, :, 4:5, ...].reshape(bs, -1, ny, nx)
        return cls_score, bbox_pred, objectness

    def forward_single(self, x: Tensor,
                       convs: nn.Module) -> Tensor:
        """Forward feature of a single scale level."""
        pred_map = convs(x)
        bs, _, ny, nx = pred_map.shape
        pred_map = pred_map.view(bs, self.num_base_priors, self.num_out_attrib,
                                 ny, nx)
        return pred_map.permute(0, 1, 3, 4, 2).contiguous()

    def process(self, pred_map) -> Tuple[Tensor,Tensor]:
        res = []

        for idx, feat_ in enumerate(pred_map):
            bs, _, ny, nx, _ = feat_.shape
            grid, grid_ = self.get_grid(nx, ny, idx, feat_.device)
            feat = feat_.sigmoid()
            xy = (feat[..., 0:2] * 2 - 0.5 + grid) * torch.as_tensor(
                self.featmap_strides[idx],
                dtype=torch.float,
                device=feat.device)
            wh = (feat[..., 2:4] * 2)**2 * grid_
            out = torch.cat((xy, wh, feat[..., 4:]), -1)
            res.append(out.view(bs, -1, self.num_out_attrib))

        return (torch.cat(res, 1), pred_map)

    def get_grid(self, x, y, idx, device):
        if torch.__version__ > '1.10.0':
            dy, dx = torch.meshgrid([
                torch.arange(y, device=device),
                torch.arange(x, device=device)
            ],
                                    indexing='ij')
        else:
            dy, dx = torch.meshgrid([
                torch.arange(y, device=device),
                torch.arange(x, device=device)
            ])
        grid = torch.stack((dx, dy), dim=2).expand(1, self.num_base_priors, y,
                                                   x, 2).float()
        grid_ = self.anchors[idx].clone().view(
            (1, self.num_base_priors, 1, 1, 2)).expand(
                (1, self.num_base_priors, y, x, 2)).float().to(device)

        return grid, grid_


@MODELS.register_module()
class YOLOV5Head(YOLOv5Head):
    head_module: DetHead

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list,
                                                               dict]) -> dict:
        outs = self.head_module(x)

        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self.head_module(x)
        predictions = self.predict_by_feat(*outs,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        return self.head_module._forward(x)