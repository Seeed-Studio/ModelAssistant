import math
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

    def _forward(self, x: Tuple[Tensor]) -> Tensor:
        assert len(x) == self.num_levels
        out = multi_apply(self.forward_single, x, self.cls_preds, self.reg_preds)
        cls_ = torch.cat(list(map(lambda x: x.permute(0, 2, 3, 1).reshape(1, -1, self.num_classes), out[0])), 1)
        bbox_ = torch.cat(list(map(lambda x: x.permute(0, 2, 3, 1).reshape(1, -1, 4), out[1])), 1)
        result = torch.cat((bbox_, cls_), -1)
        return result

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
        for i in self.mlvl_priors:
            i.to(x[0].device)
        outs = self.head_module(x)
        predictions = self.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions

    def forward(self, x: Tuple[Tensor]) -> Tensor:
        B = x[0].shape[0]
        out = super().forward(x=x)
        return self.export(*out, batch_num=B)

    def export(self, cls_scores: List[Tensor], bbox_preds: List[Tensor], batch_num: int = 1) -> Tensor:
        assert len(cls_scores) == len(bbox_preds)

        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        self.mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )
        self.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.mlvl_priors)

        mlvl_strides = [
            torch.ones((featmap_size[0] ** 2 * self.num_base_priors,)) * stride
            for featmap_size, stride in zip(featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides).to(cls_scores[0].device)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(batch_num, -1, self.num_classes) for cls_score in cls_scores
        ]
        flatten_bbox_preds = [bbox_pred.permute(0, 2, 3, 1).reshape(batch_num, -1, 4) for bbox_pred in bbox_preds]
        # In order to reduce the quantization error
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid() * 100
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_coder.decode(flatten_priors[None], flatten_bbox_preds, flatten_stride)
        result = torch.cat((flatten_decoded_bboxes, flatten_cls_scores), dim=-1)
        return result
