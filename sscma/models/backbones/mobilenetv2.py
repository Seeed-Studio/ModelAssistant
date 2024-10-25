# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch.nn as nn
from torchvision.models._utils import _make_divisible

from mmengine.model import BaseModule
from mmengine import MODELS
from sscma.models.base.general import ConvNormActivation, InvertedResidual
from sscma.models.layers.rep import RepBlock, RepConv1x1
from ..base.general import CBR, InvertedResidual


class MobileNetv2(BaseModule):
    def __init__(
        self,
        widen_factor: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[dict] = None,
        norm_layer: Optional[dict] = None,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        gray_input: bool = False,
        rep: bool = False,
        init_cfg: Union[dict, List[dict], None] = None,
    ):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        if block is None and not rep:
            block = InvertedResidual
        elif rep:
            # block = RepBlock
            block = RepConv1x1
        elif isinstance(block, dict):
            block = MODELS.build(rep)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        in_channels = 32

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        assert (
            len(inverted_residual_setting) and len(inverted_residual_setting[0]) == 4
        ), ValueError(
            f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
        )

        in_channels = _make_divisible(in_channels * widen_factor, round_nearest)

        self.conv1 = ConvNormActivation(
            1 if gray_input else 3,
            in_channels,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer="ReLU6",
        )

        self.layers = []
        for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            out_channels = _make_divisible(c * widen_factor, round_nearest) * (
                2 if rep else 1
            )
            tmp_layers = []
            for i in range(n):
                stride = s if i == 0 else 1
                if block is RepBlock:
                    layer = block(
                        in_channels,
                        out_channels,
                        stride=stride,
                        groups=in_channels,
                        norm_layer=norm_layer,
                    )

                elif block is RepConv1x1:
                    layer = block(in_channels, out_channels, stride=stride, depth=6)
                else:
                    layer = block(
                        in_channels,
                        out_channels,
                        stride,
                        expand_ratio=t,
                        norm_layer=norm_layer,
                    )
                in_channels = out_channels
                tmp_layers.append(layer)

            layer_name = f"layer{idx+1}"
            self.add_module(layer_name, nn.Sequential(*tmp_layers))
            self.layers.append(layer_name)

    def forward(self, x):
        res = []
        x = self.conv1(x)
        for idx, layer_name in enumerate(self.layers):
            x = getattr(self, layer_name)(x)
            if idx in self.out_indices:
                res.append(x)
                if idx == max(self.out_indices):
                    break
        return tuple(res)

    def init_weights(self):
        super().init_weights()
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class PfldMobileNetV2(nn.Module):
    def __init__(
        self,
        inchannel=3,
        layer1=[16, 16, 16, 16, 16],
        layer2=[32, 32, 32, 32, 32, 32],
        out_channel=16,
    ):
        super(PfldMobileNetV2, self).__init__()
        inp = 32
        self.conv1 = CBR(inchannel, inp, kernel=3, stride=2, padding=1, bias=False)
        self.conv2 = CBR(
            inp, inp, kernel=3, stride=1, padding=1, groups=inp, bias=False
        )

        layer = []
        for idx, oup in enumerate(layer1):
            if idx == 0:
                layer.append(InvertedResidual(inp, oup, 2, 2))
            else:
                layer.append(InvertedResidual(inp, oup, 1, 2))
            inp = oup
        self.layer1 = nn.Sequential(*layer)
        self.block1 = InvertedResidual(inp, 32, 2, 2)
        inp = 32

        layer = []
        for idx, oup in enumerate(layer2):
            layer.append(InvertedResidual(inp, oup, 1, 4))
            inp = oup
        self.layer2 = nn.Sequential(*layer)
        self.block2 = InvertedResidual(inp, out_channel, 1, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.layer1(x)

        x = self.block1(x)

        x = self.layer2(x)

        x = self.block2(x)

        return x
