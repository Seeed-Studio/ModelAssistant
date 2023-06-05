from typing import List, Optional, Union
import torch

import torch.nn as nn
from edgelab.registry import MODELS

from mmengine.model import BaseModule
from edgelab.models.base.general import ConvNormActivation


def padding_weights(weights, shape=(3, 3)):
    """
    Fill the convolution weights to the corresponding shape
    """
    if isinstance(shape, int):
        shape = (shape, shape)
    elif isinstance(shape, (tuple, list)):
        if len(shape) == 1:
            shape = (shape[0], shape[0])
    else:
        raise TypeError(
            'Wrong shape type, its type should be "int", "tuple", or "list",but got the type of {}'
            .format(type(shape)))

    if weights is None:
        return 0
    else:
        O, I, H, W = weights.shape
        return torch.nn.functional.pad(
            weights, ((shape[0] - W) // 2, (shape[0] - W) // 2,
                      (shape[0] - H) // 2, (shape[0] - H) // 2),
            mode='constant',
            value=0)


@MODELS.register_module()
class RepRes(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        self.kernel_size = kernel_size
        self.conv3x3 = ConvNormActivation(in_channels,
                                          out_channels,
                                          kernel_size,
                                          stride,
                                          padding,
                                          groups=groups,
                                          dilation=dilation,
                                          bias=False)
        self.conv1x1 = ConvNormActivation(in_channels,
                                          out_channels,
                                          1,
                                          groups=groups,
                                          activation_layer=None,
                                          dilation=dilation,
                                          bias=False)

        self.convbn = nn.Conv2d(in_channels,
                                out_channels,
                                3,
                                stride=stride,
                                groups=groups,
                                dilation=dilation,
                                bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        if self.training:
            self.original_forward(inputs)
            self.frist = True
        else:
            if self.frist:
                self.rep()
                self.frist = False
            self.rep_forward(inputs)

    def rep(self):
        k3x3, b3x3 = self.fuse_conv_bn(self.conv3x3)
        k1x1, b1x1 = self.fuse_conv_bn(self.conv1x1)
        weights, bias = k3x3 + padding_weights(k1x1,
                                               self.kernel_size), b3x3 + b1x1
        self.convbn.weight.data = weights
        self.convbn.bias.data = bias

    def fuse_conv_bn(self, block):
        if isinstance(block, nn.Sequential):
            block: ConvNormActivation
            kernel, nor_mean, nor_var, nor_weights, nor_biase, nor_eps = block.conv.weight, block.norm.running_mean, block.norm.running_var, block.weights, block.bias, block.eps
        elif isinstance(block, nn.Identity):
            return 0, 0

        std = (nor_var + nor_eps).sqrt()
        t = (nor_weights / std).reshape(-1, 1, 1, 1)
        return kernel * t, nor_weights - nor_mean * nor_biase / std

    def original_forward(self, inputs):
        return self.conv3x3(inputs) + self.conv1x1(inputs) + inputs

    def rep_forward(self, inputs):
        return self.convbn(inputs)


@MODELS.register_module()
class RepLine(BaseModule):

    def __init__(self, init_cfg: Union[dict, List[dict], None] = None):
        super().__init__(init_cfg)
