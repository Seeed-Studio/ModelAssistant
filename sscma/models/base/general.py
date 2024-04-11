# copyright Copyright (c) Seeed Technology Co.,Ltd.
from typing import Any, AnyStr, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm


def get_conv(conv):
    if isinstance(conv, dict) and hasattr(nn, conv.get('type')):
        conv = getattr(nn, conv.get('type'))
    elif isinstance(conv, str) and hasattr(nn, conv):
        conv = getattr(nn, conv)
    elif isinstance(conv, str) and conv in MODELS.module_dict:
        conv = MODELS.get(conv)
    elif (isinstance(conv, type.__class__) and issubclass(conv, nn.Module)) or hasattr(conv, '__call__'):
        pass
    else:
        raise ValueError(
            'Unable to parse the value of conv_layer, please confirm whether the value of conv_layer is correct'
        )
    return conv


def get_norm(norm):
    if isinstance(norm, dict) and hasattr(nn, norm['type']):
        norm = getattr(nn, norm.get('type'))
    elif isinstance(norm, dict) and norm.get('type') in MODELS.module_dict:
        norm = MODELS.get(norm.get('type'))
    elif isinstance(norm, str) and hasattr(nn, norm):
        norm = getattr(nn, norm)
    elif isinstance(norm, str) and norm in MODELS.module_dict:
        norm = MODELS.get(norm)
    elif (isinstance(norm, type.__class__) and issubclass(norm, nn.Module)) or hasattr(norm, '__call__'):
        pass
    else:
        raise ValueError(
            'Unable to parse the value of norm_layer, please confirm whether the value of norm_layer is correct'
        )
    return norm


def get_act(act):
    if isinstance(act, dict) and hasattr(nn, act.get('type')):
        act = getattr(nn, act.get('type'))
    elif isinstance(act, str) and hasattr(nn, act):
        act = getattr(nn, act)
    elif isinstance(act, str) and act in MODELS.module_dict:
        act = MODELS.get(act)
    elif (isinstance(act, type.__class__) and issubclass(act, nn.Module)) or hasattr(act, '__call__'):
        pass
    else:
        raise ValueError(
            'Unable to parse the value of act_layer, please confirm whether the value of act_layer is correct'
        )
    return act


def is_norm(layer: nn.Module, exclude: Union[type, tuple, None] = None) -> bool:
    all_norm_bases = (_BatchNorm, _InstanceNorm, nn.GroupNorm, nn.LayerNorm)
    return isinstance(layer, all_norm_bases)


class ConvNormActivation(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        bias: Optional[bool] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] or Dict or AnyStr = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] or Dict or AnyStr = nn.ReLU,
        conv_layer: Optional[Callable[..., nn.Module]] or Dict or AnyStr = None,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if conv_layer is None:
            conv_layer = nn.Conv2d
        else:
            conv_layer = get_conv(conv_layer)
        conv = conv_layer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
            bias=norm_layer is None if bias is None else bias,
        )
        self.add_module('conv', conv)
        if norm_layer is not None:
            norm_layer = get_norm(norm_layer)
            self.add_module('norm', norm_layer(out_channels))
        if activation_layer is not None:
            activation_layer = get_act(activation_layer)
            self.add_module('act', activation_layer())

        self.out_channels = out_channels


class SqueezeExcitation(torch.nn.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Any = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.activation = get_act(activation)(inplace=True)
        self.conv1 = ConvNormActivation(
            input_channels, squeeze_channels, 1, padding=0, norm_layer=None, activation_layer=activation
        )
        self.conv2 = ConvNormActivation(
            squeeze_channels, input_channels, 1, padding=0, norm_layer=None, activation_layer=activation
        )
        self.scale_activation = get_act(scale_activation)()

    def _scale(self, input: torch.Tensor) -> torch.Tensor:
        scale = self.avgpool(input)
        scale = self.conv1(scale)
        scale = self.conv2(scale)
        return self.scale_activation(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input)
        return scale * input


def CBR(inp, oup, kernel, stride, bias=False, padding=1, groups=1, act='ReLU'):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, groups=groups, bias=bias),
        nn.BatchNorm2d(oup),
        nn.Identity() if not act else getattr(nn, act)(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f'stride should be 1 or 2 instead of {stride}')

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
            )
        layers.extend(
            [
                # dw
                ConvNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=norm_layer,
                    activation_layer=nn.ReLU6,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
