from typing import List, Union, Optional, Tuple

import torch.nn as nn

from mmengine.model import BaseModule
from edgelab.registry import BACKBONES, MODELS
from torchvision.models._utils import _make_divisible
from edgelab.models.layers.rep import RepBlock
from ..base.general import InvertedResidual, ConvNormActivation


@BACKBONES.register_module()
class MobileNetv2(BaseModule):

    def __init__(self,
                 widen_factor: float = 1.0,
                 inverted_residual_setting: Optional[List[List[int]]] = None,
                 round_nearest: int = 8,
                 block: Optional[dict] = None,
                 norm_layer: Optional[dict] = None,
                 out_indices: Tuple[int, ...] = (1, 2, 3),
                 gray_input: bool = False,
                 rep: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        if block is None and not rep:
            block = InvertedResidual
        elif rep:
            block = RepBlock
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

        assert len(inverted_residual_setting) and len(
            inverted_residual_setting[0]
        ) == 4, ValueError(
            f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
        )

        in_channels = _make_divisible(in_channels * widen_factor,
                                      round_nearest)

        self.conv1 = ConvNormActivation(1 if gray_input else 3,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        norm_layer=norm_layer,
                                        activation_layer='ReLU6')

        self.layers = []
        for idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            out_channels = _make_divisible(c * widen_factor, round_nearest)
            tmp_layers = []
            for i in range(n):
                stride = s if i == 0 else 1
                if block is RepBlock:
                    layer = block(in_channels,
                                  out_channels,
                                  stride=stride,
                                  groups=in_channels,
                                  norm_layer=norm_layer)
                else:
                    layer = block(in_channels,
                                  out_channels,
                                  stride,
                                  expand_ratio=t)
                in_channels = out_channels
                tmp_layers.append(layer)

            layer_name = f'layer{idx+1}'
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