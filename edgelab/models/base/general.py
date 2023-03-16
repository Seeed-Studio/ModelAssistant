from typing import Optional, Callable, Dict, AnyStr
import torch.nn as nn
from mmcv.cnn.bricks.norm import NORM_LAYERS
from mmcv.cnn.bricks.activation import ACTIVATION_LAYERS
from mmcv.cnn.bricks.conv import CONV_LAYERS


def get_conv(conv):
    if isinstance(conv, dict) and hasattr(nn, conv.get('type')):
        conv = getattr(nn, conv.get('type'))
    elif isinstance(conv, str) and hasattr(nn, conv):
        conv = getattr(nn, conv)
    elif isinstance(conv, str) and conv in CONV_LAYERS.module_dict:
        conv = CONV_LAYERS.get(conv)
    elif (isinstance(conv, type.__class__)
          and issubclass(conv, nn.Module)) or hasattr(conv, '__call__'):
        pass
    else:
        raise ValueError(
            'Unable to parse the value of conv_layer, please confirm whether the value of conv_layer is correct'
        )
    return conv


def get_norm(norm):
    if isinstance(norm, dict) and hasattr(nn, norm['type']):
        norm = getattr(nn, norm.get('type'))
    elif isinstance(norm,
                    dict) and norm.get('type') in NORM_LAYERS.module_dict:
        norm = NORM_LAYERS.get(norm.get('type'))
    elif isinstance(norm, str) and hasattr(nn, norm):
        norm = getattr(nn, norm)
    elif isinstance(norm, str) and norm in NORM_LAYERS.module_dict:
        norm = NORM_LAYERS.get(norm)
    elif (isinstance(norm, type.__class__)
          and issubclass(norm, nn.Module)) or hasattr(norm, '__call__'):
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
    elif isinstance(act, str) and act in ACTIVATION_LAYERS.module_dict:
        act = ACTIVATION_LAYERS.get(act)
    elif (isinstance(act, type.__class__)
          and issubclass(act, nn.Module)) or hasattr(act, '__call__'):
        pass
    else:
        raise ValueError(
            'Unable to parse the value of act_layer, please confirm whether the value of act_layer is correct'
        )
    return act


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
        norm_layer: Optional[Callable[..., nn.Module]] or Dict
        or AnyStr = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] or Dict
        or AnyStr = nn.ReLU,
        conv_layer: Optional[Callable[..., nn.Module]] or Dict
        or AnyStr = None,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if conv_layer is None:
            conv_layer = nn.Conv2d
        else:
            conv_layer = get_conv(conv_layer)
        layers = [
            conv_layer(in_channels,
                       out_channels,
                       kernel_size,
                       stride,
                       padding,
                       dilation=dilation,
                       groups=groups,
                       bias=norm_layer is None if bias is None else bias)
        ]
        if norm_layer is not None:
            norm_layer = get_norm(norm_layer)
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            activation_layer = get_act(activation_layer)
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels


def CBR(inp, oup, kernel, stride, bias=False, padding=1, groups=1, act='ReLU'):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, groups=groups, bias=bias),
        nn.BatchNorm2d(oup),
        nn.Identity() if not act else getattr(nn, act)(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, residual, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.residual = residual

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio,
                      inp * expand_ratio,
                      3,
                      stride,
                      1,
                      groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.residual:
            return x + self.conv(x)
        else:
            return self.conv(x)