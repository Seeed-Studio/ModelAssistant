# Copyright (c) OpenMMLab. All rights reserved.
import math
from abc import ABCMeta, abstractmethod
from typing import Sequence, Union, List

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.model import BaseModule
from sscma.utils.typing_utils import ConfigType, OptMultiConfig
from ..layers import CSPLayer
from ..cnn import ConvModule, DepthwiseSeparableConvModule


class BaseYOLONeck(BaseModule, metaclass=ABCMeta):
    """Base neck used in YOLO series.

    .. code:: text

     P5 neck model structure diagram
                        +--------+                     +-------+
                        |top_down|----------+--------->|  out  |---> output0
                        | layer1 |          |          | layer0|
                        +--------+          |          +-------+
     stride=8                ^              |
     idx=0  +------+    +--------+          |
     -----> |reduce|--->|   cat  |          |
            |layer0|    +--------+          |
            +------+         ^              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer1 |    |  layer0   |
                        +--------+    +-----------+
                             ^              |
                        +--------+          v
                        |top_down|    +-----------+
                        | layer2 |--->|    cat    |
                        +--------+    +-----------+
     stride=16               ^              v
     idx=1  +------+    +--------+    +-----------+    +-------+
     -----> |reduce|--->|   cat  |    | bottom_up |--->|  out  |---> output1
            |layer1|    +--------+    |   layer0  |    | layer1|
            +------+         ^        +-----------+    +-------+
                             |              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer2 |    |  layer1   |
     stride=32          +--------+    +-----------+
     idx=2  +------+         ^              v
     -----> |reduce|         |        +-----------+
            |layer2|---------+------->|    cat    |
            +------+                  +-----------+
                                            v
                                      +-----------+    +-------+
                                      | bottom_up |--->|  out  |---> output2
                                      |  layer1   |    | layer2|
                                      +-----------+    +-------+

    .. code:: text

     P6 neck model structure diagram
                        +--------+                     +-------+
                        |top_down|----------+--------->|  out  |---> output0
                        | layer1 |          |          | layer0|
                        +--------+          |          +-------+
     stride=8                ^              |
     idx=0  +------+    +--------+          |
     -----> |reduce|--->|   cat  |          |
            |layer0|    +--------+          |
            +------+         ^              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer1 |    |  layer0   |
                        +--------+    +-----------+
                             ^              |
                        +--------+          v
                        |top_down|    +-----------+
                        | layer2 |--->|    cat    |
                        +--------+    +-----------+
     stride=16               ^              v
     idx=1  +------+    +--------+    +-----------+    +-------+
     -----> |reduce|--->|   cat  |    | bottom_up |--->|  out  |---> output1
            |layer1|    +--------+    |   layer0  |    | layer1|
            +------+         ^        +-----------+    +-------+
                             |              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer2 |    |  layer1   |
                        +--------+    +-----------+
                             ^              |
                        +--------+          v
                        |top_down|    +-----------+
                        | layer3 |--->|    cat    |
                        +--------+    +-----------+
     stride=32               ^              v
     idx=2  +------+    +--------+    +-----------+    +-------+
     -----> |reduce|--->|   cat  |    | bottom_up |--->|  out  |---> output2
            |layer2|    +--------+    |   layer1  |    | layer2|
            +------+         ^        +-----------+    +-------+
                             |              v
                        +--------+    +-----------+
                        |upsample|    |downsample |
                        | layer3 |    |  layer2   |
                        +--------+    +-----------+
     stride=64               ^              v
     idx=3  +------+         |        +-----------+
     -----> |reduce|---------+------->|    cat    |
            |layer3|                  +-----------+
            +------+                        v
                                      +-----------+    +-------+
                                      | bottom_up |--->|  out  |---> output3
                                      |  layer2   |    | layer3|
                                      +-----------+    +-------+

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        upsample_feats_cat_first (bool): Whether the output features are
            concat first after upsampling in the topdown module.
            Defaults to True. Currently only YOLOv7 is false.
        freeze_all(bool): Whether to freeze the model. Defaults to False
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to None.
        act_cfg (dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: Union[int, List[int]],
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        upsample_feats_cat_first: bool = True,
        freeze_all: bool = False,
        norm_cfg: ConfigType = None,
        act_cfg: ConfigType = None,
        init_cfg: OptMultiConfig = None,
        **kwargs
    ):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deepen_factor = deepen_factor
        self.widen_factor = widen_factor
        self.upsample_feats_cat_first = upsample_feats_cat_first
        self.freeze_all = freeze_all
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.reduce_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.reduce_layers.append(self.build_reduce_layer(idx))

        # build top-down blocks
        self.upsample_layers = nn.ModuleList()
        self.top_down_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1, 0, -1):
            self.upsample_layers.append(self.build_upsample_layer(idx))
            self.top_down_layers.append(self.build_top_down_layer(idx))

        # build bottom-up blocks
        self.downsample_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()
        for idx in range(len(in_channels) - 1):
            self.downsample_layers.append(self.build_downsample_layer(idx))
            self.bottom_up_layers.append(self.build_bottom_up_layer(idx))

        self.out_layers = nn.ModuleList()
        for idx in range(len(in_channels)):
            self.out_layers.append(self.build_out_layer(idx))

    @abstractmethod
    def build_reduce_layer(self, idx: int):
        """build reduce layer."""
        pass

    @abstractmethod
    def build_upsample_layer(self, idx: int):
        """build upsample layer."""
        pass

    @abstractmethod
    def build_top_down_layer(self, idx: int):
        """build top down layer."""
        pass

    @abstractmethod
    def build_downsample_layer(self, idx: int):
        """build downsample layer."""
        pass

    @abstractmethod
    def build_bottom_up_layer(self, idx: int):
        """build bottom up layer."""
        pass

    @abstractmethod
    def build_out_layer(self, idx: int):
        """build out layer."""
        pass

    def _freeze_all(self):
        """Freeze the model."""
        for m in self.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep the normalization
        layer freezed."""
        super().train(mode)
        if self.freeze_all:
            self._freeze_all()

    def forward(self, inputs: List[torch.Tensor]) -> tuple:
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # reduce layers
        reduce_outs = []
        for idx in range(len(self.in_channels)):
            reduce_outs.append(self.reduce_layers[idx](inputs[idx]))

        # top-down path
        inner_outs = [reduce_outs[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = reduce_outs[idx - 1]
            upsample_feat = self.upsample_layers[len(self.in_channels) - 1 - idx](
                feat_high
            )
            if self.upsample_feats_cat_first:
                top_down_layer_inputs = torch.cat([upsample_feat, feat_low], 1)
            else:
                top_down_layer_inputs = torch.cat([feat_low, upsample_feat], 1)
            inner_out = self.top_down_layers[len(self.in_channels) - 1 - idx](
                top_down_layer_inputs
            )
            inner_outs.insert(0, inner_out)

        # bottom-up path
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_layers[idx](feat_low)
            out = self.bottom_up_layers[idx](torch.cat([downsample_feat, feat_high], 1))
            outs.append(out)

        # out_layers
        results = []
        for idx in range(len(self.in_channels)):
            results.append(self.out_layers[idx](outs[idx]))

        return tuple(results)


class CSPNeXtPAFPN(BaseYOLONeck):
    """Path Aggregation Network with CSPNeXt blocks.

    Args:
        in_channels (Sequence[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        deepen_factor (float): Depth multiplier, multiply number of
            blocks in CSP layer by this amount. Defaults to 1.0.
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Defaults to 1.0.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer.
            Defaults to 3.
        use_depthwise (bool): Whether to use depthwise separable convolution in
            blocks. Defaults to False.
        expand_ratio (float): Ratio to adjust the number of channels of the
            hidden layer. Defaults to 0.5.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='SiLU', inplace=True)
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        num_csp_blocks: int = 3,
        freeze_all: bool = False,
        use_depthwise: bool = False,
        expand_ratio: float = 0.5,
        upsample_cfg: ConfigType = dict(scale_factor=2, mode="nearest"),
        conv_cfg: bool = None,
        norm_cfg: ConfigType = dict(type="BN"),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = dict(
            type="Kaiming",
            layer="Conv2d",
            a=math.sqrt(5),
            distribution="uniform",
            mode="fan_in",
            nonlinearity="leaky_relu",
        ),
    ) -> None:
        self.num_csp_blocks = round(num_csp_blocks * deepen_factor)
        self.conv = DepthwiseSeparableConvModule if use_depthwise else ConvModule
        self.upsample_cfg = upsample_cfg
        self.expand_ratio = expand_ratio
        self.conv_cfg = conv_cfg

        super().__init__(
            in_channels=[int(channel * widen_factor) for channel in in_channels],
            out_channels=int(out_channels * widen_factor),
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            freeze_all=freeze_all,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg,
        )

    def build_reduce_layer(self, idx: int) -> nn.Module:
        """build reduce layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The reduce layer.
        """
        if idx == len(self.in_channels) - 1:
            layer = self.conv(
                self.in_channels[idx],
                self.in_channels[idx - 1],
                1,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            layer = nn.Identity()

        return layer

    def build_upsample_layer(self, *args, **kwargs) -> nn.Module:
        """build upsample layer."""
        return nn.Upsample(**self.upsample_cfg)

    def build_top_down_layer(self, idx: int) -> nn.Module:
        """build top down layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The top down layer.
        """
        if idx == 1:
            return CSPLayer(
                self.in_channels[idx - 1] * 2,
                self.in_channels[idx - 1],
                num_blocks=self.num_csp_blocks,
                add_identity=False,
                use_cspnext_block=True,
                expand_ratio=self.expand_ratio,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
        else:
            return nn.Sequential(
                CSPLayer(
                    self.in_channels[idx - 1] * 2,
                    self.in_channels[idx - 1],
                    num_blocks=self.num_csp_blocks,
                    add_identity=False,
                    use_cspnext_block=True,
                    expand_ratio=self.expand_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                self.conv(
                    self.in_channels[idx - 1],
                    self.in_channels[idx - 2],
                    kernel_size=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
            )

    def build_downsample_layer(self, idx: int) -> nn.Module:
        """build downsample layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The downsample layer.
        """
        return self.conv(
            self.in_channels[idx],
            self.in_channels[idx],
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_bottom_up_layer(self, idx: int) -> nn.Module:
        """build bottom up layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The bottom up layer.
        """
        return CSPLayer(
            self.in_channels[idx] * 2,
            self.in_channels[idx + 1],
            num_blocks=self.num_csp_blocks,
            add_identity=False,
            use_cspnext_block=True,
            expand_ratio=self.expand_ratio,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_out_layer(self, idx: int) -> nn.Module:
        """build out layer.

        Args:
            idx (int): layer idx.

        Returns:
            nn.Module: The out layer.
        """
        return self.conv(
            self.in_channels[idx],
            self.out_channels,
            3,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
