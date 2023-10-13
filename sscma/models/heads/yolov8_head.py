import math
from functools import partial
from typing import List, Sequence, Tuple, Union
from mmdet.structures import SampleList

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.utils import multi_apply
from mmdet.utils import ConfigType, InstanceList, OptMultiConfig

from mmengine.model import BaseModule
from torch import Tensor

from sscma.registry import MODELS
from mmyolo.models.utils import make_divisible

from mmyolo.models.dense_heads import YOLOv8Head
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead


@MODELS.register_module(name='CusYOLOv8HeadModule')
class YOLOv8HeadModule(BaseModule):
    """YOLOv8HeadModule head module used in `YOLOv8`.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (Union[int, Sequence]): Number of channels in the input
            feature map.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_base_priors (int): The number of priors (points) at a point
            on the feature grid.
        featmap_strides (Sequence[int]): Downsample factor of each feature map.
             Defaults to [8, 16, 32].
        reg_max (int): Max value of integral set :math: ``{0, ..., reg_max-1}``
            in QFL setting. Defaults to 16.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: Union[int, Sequence],
        widen_factor: float = 1.0,
        num_base_priors: int = 1,
        featmap_strides: Sequence[int] = (8, 16, 32),
        reg_max: int = 16,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU', inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.num_base_priors = num_base_priors
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_channels = in_channels
        self.reg_max = reg_max

        in_channels = []
        for channel in self.in_channels:
            channel = make_divisible(channel, widen_factor)
            in_channels.append(channel)
        self.in_channels = in_channels

        self._init_layers()

    def init_weights(self, prior_prob=0.01):
        """Initialize the weight and bias of PPYOLOE head."""
        super().init_weights()
        for reg_pred, cls_pred, stride in zip(self.reg_preds, self.cls_preds, self.featmap_strides):
            reg_pred[-1].bias.data[:] = 1.0  # box
            # cls (.01 objects, 80 classes, 640 img)
            cls_pred[-1].bias.data[: self.num_classes] = math.log(5 / self.num_classes / (640 / stride) ** 2)

    def _init_layers(self):
        """initialize conv layers in YOLOv8 head."""
        # Init decouple head
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        reg_out_channels = max((16, self.in_channels[0] // 4, self.reg_max * 4))
        cls_out_channels = max(self.in_channels[0], self.num_classes)

        for i in range(self.num_levels):
            self.reg_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    ConvModule(
                        in_channels=reg_out_channels,
                        out_channels=reg_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    nn.Conv2d(in_channels=reg_out_channels, out_channels=4 * self.reg_max, kernel_size=1),
                )
            )
            self.cls_preds.append(
                nn.Sequential(
                    ConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    ConvModule(
                        in_channels=cls_out_channels,
                        out_channels=cls_out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    ),
                    nn.Conv2d(in_channels=cls_out_channels, out_channels=self.num_classes, kernel_size=1),
                )
            )

        proj = torch.arange(self.reg_max, dtype=torch.float)
        self.register_buffer('proj', proj, persistent=False)

    def forward(self, x: Tuple[Tensor]) -> Tuple[List]:
        """Forward features from the upstream network.

        Args:
            x (Tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            Tuple[List]: A tuple of multi-level classification scores, bbox
            predictions
        """
        assert len(x) == self.num_levels
        return multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)

    def forward_single(self, x: torch.Tensor, cls_pred: nn.ModuleList, reg_pred: nn.ModuleList) -> Tuple:
        """Forward feature of a single scale level."""
        b, _, h, w = x.shape
        cls_logit = cls_pred(x)
        bbox_dist_preds = reg_pred(x)
        if self.reg_max > 1:
            bbox_dist_preds = bbox_dist_preds.reshape([-1, 4, self.reg_max, h * w]).permute(0, 3, 1, 2)
            # TODO: The get_flops script cannot handle the situation of
            #  matmul, and needs to be fixed later
            # bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj)
            bbox_preds = bbox_dist_preds.softmax(3).matmul(self.proj.view([-1, 1])).squeeze(-1)
            bbox_preds = bbox_preds.transpose(1, 2).reshape(b, -1, h, w)
        else:
            bbox_preds = bbox_dist_preds
        if self.training:
            return cls_logit, bbox_preds, bbox_dist_preds
        else:
            return cls_logit, bbox_preds


@MODELS.register_module(name='CusYOLOv8Head')
class YOLOv8Head(YOLOv8Head):
    head_module: YOLOv8HeadModule

    def loss(self, x: Tuple[Tensor], batch_data_samples: Union[list, dict]) -> dict:
        if isinstance(batch_data_samples, list):
            losses = BaseDenseHead.loss(self, x, batch_data_samples)
        else:
            outs = self.head_module(x)
            # Fast version
            loss_inputs = outs + (batch_data_samples['bboxes_labels'], batch_data_samples['img_metas'])
            losses = self.loss_by_feat(*loss_inputs)

        return losses

    def predict(self, x: Tuple[Tensor], batch_data_samples: SampleList, rescale: bool = False) -> InstanceList:
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        outs = self.head_module(x)
        predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        B = x[0].shape[0]
        out = super().forward(x=x)
        return self.export(*out, batch_num=B)

    def export(self, cls_scores: List[Tensor], bbox_preds: List[Tensor], batch_num: int = 1) -> Tensor:
        assert len(cls_scores) == len(bbox_preds)
        featmap_len = len(cls_scores)
        tmp = [torch.cat((bbox_preds[idx], cls_scores[idx]), 1) for idx in range(featmap_len)]
        anchors, stides = (i.transpose(0, 1) for i in make_anchors(tmp, self.head_module.featmap_strides, 0.5))
        x_cat = torch.cat([i.reshape(batch_num, 4 + self.num_classes, -1) for i in tmp], 2)
        bbox, cls = x_cat.split((4, self.num_classes), 1)
        lt, rb = bbox.chunk(2, 1)
        x1y1 = anchors.unsqueeze(0) - lt
        x2y2 = anchors.unsqueeze(0) + rb
        dbox = torch.cat((x1y1, x2y2), 1) * stides
        preds = torch.cat((dbox.permute(0, 2, 1), cls.sigmoid().permute(0, 2, 1) * 100), 2)
        return preds


def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = feats[i].shape
        sx = torch.arange(0, w, device=device) + grid_cell_offset  # shift x
        sy = torch.arange(0, h, device=device) + grid_cell_offset  # shift y
        sx = sx.to(dtype=dtype)
        sy = sy.to(dtype=dtype)
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') if torch.__version__ == '1.10.0' else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((w * h, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)
