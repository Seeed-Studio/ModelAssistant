# copyright Copyright (c) Seeed Technology Co.,Ltd.
from typing import List, Tuple, Union

import torch.nn as nn
from mmengine.model.base_module import BaseModule

from sscma.models.base.general import ConvNormActivation
from sscma.registry import MODELS


@MODELS.register_module()
class SEAttention(BaseModule):
    def __init__(self, in_channels: int, r: int = 4, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        middle_channels = in_channels // r
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = ConvNormActivation(in_channels, middle_channels, 1, 1, bias=True, activation_layer='ReLU')
        self.conv2 = ConvNormActivation(middle_channels, in_channels, 1, 1, bias=True, activation_layer='Sigmoid')

    def forward(self, inputs):
        x = self.conv2(self.conv1(self.avgPool(inputs)))
        return x * inputs


@MODELS.register_module()
class SpatialAttention(BaseModule):
    def __init__(self, kernel_size: int = 3, stride: int = 1, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            pass
        else:
            raise ValueError()

        self.conv = ConvNormActivation(
            2,
            1,
            kernel_size,
            stride,
            padding=((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2),
            activation_layer='Sigmoid',
        )

    def forward(self, inputs):
        max_x, _ = torch.max(inputs, dim=1, keepdim=True)
        mean_x = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.concat((max_x, mean_x), dim=1)
        x = self.conv(x)
        return x * inputs


@MODELS.register_module()
class ChannelAttention(BaseModule):
    def __init__(self, in_channels: int, r: int = 4, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxPool = nn.AdaptiveMaxPool2d((1, 1))

        self.conv1 = ConvNormActivation(in_channels, in_channels // 2, 1, bias=False, activation_layer='ReLU')
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        avg_x = self.conv2(self.conv1(self.avgPool(inputs)))
        max_x = self.conv2(self.conv1(self.maxPool(inputs)))
        x = avg_x + max_x
        x = self.act(x)

        return x * inputs


@MODELS.register_module()
class CBAMAttention(BaseModule):
    def __init__(
        self, in_channels: int, kernel_size: int = 3, r: int = 4, init_cfg: Union[dict, List[dict], None] = None
    ):
        super().__init__(init_cfg)
        self.ca = ChannelAttention(in_channels, r=r)
        self.sa = SpatialAttention()

    def forward(self, inputs):
        residual = inputs
        x = self.ca(inputs)
        x = self.sa(x)
        return residual + x


@MODELS.register_module()
class SERes(BaseModule):
    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)


@MODELS.register_module()
class CA(BaseModule):
    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)


@MODELS.register_module()
class ECAMAttention(BaseModule):
    def __init__(self, in_channels: int, kernel_size: int = 3, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        x: torch.Tensor = self.avgPool(inputs)
        x = self.conv(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x = self.act(x)
        return inputs * x.expand_as(inputs)


@MODELS.register_module()
class ECA(BaseModule):
    def __init__(
        self, in_channels: int, kernel_size: Union[int, Tuple[int]] = 3, init_cfg: Union[dict, List[dict], None] = None
    ):
        super().__init__(init_cfg)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif isinstance(kernel_size, tuple):
            pass
        else:
            raise ValueError()

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size[1]), padding=(0, (kernel_size[1] - 1) // 2))

        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size[1], bias=False, groups=in_channels)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        x: torch.Tensor = self.avgPool(inputs)
        x = self.unfold(x.transpose(-1, -3))

        x = self.conv(x.transpose(-1, -2)).unsqueeze(-1)
        x = self.act(x)
        return inputs * x.expand_as(inputs)


if __name__ == '__main__':
    import torch

    se = CBAMAttention(32, 3, 4)
    input = torch.rand((16, 32, 192, 192))
    print(se(input).shape)
