# Copyright (c) Seeed Tech Ltd. All rights reserved.
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .general import ConvNormActivation


class ConvModule:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, conv_cfg, norm_cfg, act_cfg, groups=1):
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv = ConvNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            conv_layer=conv_cfg,
            norm_layer=norm_cfg,
            groups=groups,
            activation_layer=act_cfg,
        )
        return self.conv


class DepthwiseSeparableConvModule(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        norm_cfg: Optional[Dict] = None,
        act_cfg: Dict = dict(type='ReLU'),
        dw_norm_cfg: Union[Dict, str] = 'default',
        dw_act_cfg: Union[Dict, str] = 'default',
        pw_norm_cfg: Union[Dict, str] = 'default',
        pw_act_cfg: Union[Dict, str] = 'default',
        **kwargs,
    ):
        super().__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        self.depthwise_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            norm_cfg=dw_norm_cfg,  # type: ignore
            act_cfg=dw_act_cfg,  # type: ignore
            **kwargs,
        )

        self.pointwise_conv = ConvModule(
            in_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,  # type: ignore
            act_cfg=pw_act_cfg,  # type: ignore
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
