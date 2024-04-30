# Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) OpenMMLab.
import torch
import torch.nn as nn
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor

from sscma.models.base import ConvNormActivation


class ChannelAttention(BaseModule):
    def __init__(self, channels: int, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function for ChannelAttention."""
        with torch.cuda.amp.autocast(enabled=False):
            out = self.global_avgpool(x)
        out = self.fc(out)
        out = self.act(out)
        return x * out


class CSPLayer(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 0.5,
        num_blocks: int = 1,
        add_identity: bool = True,
        use_depthwise: bool = False,
        use_cspnext_block: bool = False,
        channel_attention: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='Swish'),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        block = CSPNeXtBlock if use_cspnext_block else DarknetBottleneck
        mid_channels = int(out_channels * expand_ratio)
        self.channel_attention = channel_attention
        self.main_conv = ConvNormActivation(
            in_channels, mid_channels, 1, conv_layer=conv_cfg, norm_layer=norm_cfg, activation_layer=act_cfg
        )
        self.short_conv = ConvNormActivation(
            in_channels, mid_channels, 1, conv_layer=conv_cfg, norm_layer=norm_cfg, activation_layer=act_cfg
        )
        self.final_conv = ConvNormActivation(
            2 * mid_channels, out_channels, 1, conv_layer=conv_cfg, norm_layer=norm_cfg, activation_layer=act_cfg
        )

        self.blocks = nn.Sequential(
            *[
                block(
                    mid_channels,
                    mid_channels,
                    1.0,
                    add_identity,
                    use_depthwise,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
                for _ in range(num_blocks)
            ]
        )
        if channel_attention:
            self.attention = ChannelAttention(2 * mid_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        x_short = self.short_conv(x)

        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)

        x_final = torch.cat((x_main, x_short), dim=1)

        if self.channel_attention:
            x_final = self.attention(x_final)
        return self.final_conv(x_final)


class DarknetBottleneck(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='Swish'),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormActivation(
            in_channels,
            hidden_channels,
            1,
            conv_layer=conv_cfg,
            norm_layer=norm_cfg,
            activation_layer=act_cfg,
            use_depthwise=False,
        )
        self.conv2 = ConvNormActivation(
            hidden_channels,
            out_channels,
            3,
            stride=1,
            padding=1,
            conv_layer=conv_cfg,
            norm_layer=norm_cfg,
            activation_layer=act_cfg,
            use_depthwise=use_depthwise,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out


class CSPNeXtBlock(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: float = 0.5,
        add_identity: bool = True,
        use_depthwise: bool = False,
        kernel_size: int = 5,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type='SiLU'),
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormActivation(
            in_channels,
            hidden_channels,
            3,
            stride=1,
            padding=1,
            norm_layer=norm_cfg,
            activation_layer=act_cfg,
            use_depthwise=use_depthwise,
        )
        self.conv2 = ConvNormActivation(
            hidden_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_layer=conv_cfg,
            norm_layer=norm_cfg,
            activation_layer=act_cfg,
            use_depthwise=True,
        )
        self.add_identity = add_identity and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.add_identity:
            return out + identity
        else:
            return out
