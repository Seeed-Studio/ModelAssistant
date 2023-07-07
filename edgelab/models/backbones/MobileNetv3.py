from typing import List, Callable
from functools import partial
import torch.nn as nn
from torch import Tensor

from mmengine.model import BaseModule
from mmdet.registry import VISBACKENDS
from mmdet.models.utils.make_divisible import make_divisible
from edgelab.models.base.general import ConvNormActivation, get_norm
from torchvision.ops.misc import SqueezeExcitation as SElayer


class InvertedResidualConfig:
    # Analytic model configuration table
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        widen_factor: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, widen_factor)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, widen_factor)
        self.out_channels = self.adjust_channels(out_channels, widen_factor)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, widen_factor: float):
        return make_divisible(channels * widen_factor, 8)


class InvertedResidual(nn.Module):
    # Main details of MobileNetV3
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


@VISBACKENDS.register_module()
class MobileNetV3(BaseModule):
    def __init__(
        self,
        arch='small',
        widen_factor=1,
        out_indices=(1,),
        frozen_stages=-1,
        input_channels: int = 3,
        conv_cfg=dict(type='Conv'),
        norm_cfg=None,
        act_cfg=dict(type='Hardswish'),
        norm_eval=False,
        reduced_tail: bool = False,
        dilated: bool = False,
        pretrained=None,
        init_cfg=None,
        **kwargs,
    ):
        super(MobileNetV3, self).__init__(init_cfg)

        reduce_divider = 2 if reduced_tail else 1
        dilation = 2 if dilated else 1
        ir_conf = partial(InvertedResidualConfig, widen_factor=widen_factor)

        if arch == "large":
            inverted_residual_setting = [
                ir_conf(16, 3, 16, 16, False, "RE", 1, 1),
                ir_conf(16, 3, 64, 24, False, "RE", 2, 1),  # 1/4
                ir_conf(24, 3, 72, 24, False, "RE", 1, 1),
                ir_conf(24, 5, 72, 40, True, "RE", 2, 1),  # 1/8
                ir_conf(40, 5, 120, 40, True, "RE", 1, 1),
                ir_conf(40, 5, 120, 40, True, "RE", 1, 1),
                ir_conf(40, 3, 240, 80, False, "HS", 2, 1),  # 1/16
                ir_conf(80, 3, 200, 80, False, "HS", 1, 1),
                ir_conf(80, 3, 184, 80, False, "HS", 1, 1),
                ir_conf(80, 3, 184, 80, False, "HS", 1, 1),
                ir_conf(80, 3, 480, 112, True, "HS", 1, 1),
                ir_conf(112, 3, 672, 112, True, "HS", 1, 1),
                ir_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
                ir_conf(
                    160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation
                ),
                ir_conf(
                    160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation
                ),
            ]
        elif arch == "small":
            inverted_residual_setting = [
                ir_conf(16, 3, 16, 16, True, "RE", 2, 1),  # 1/4
                ir_conf(16, 3, 72, 24, False, "RE", 2, 1),  # 1/8
                ir_conf(24, 3, 88, 24, False, "RE", 1, 1),
                ir_conf(24, 5, 96, 40, True, "HS", 2, 1),  # 1/16
                ir_conf(40, 5, 240, 40, True, "HS", 1, 1),
                ir_conf(40, 5, 240, 40, True, "HS", 1, 1),
                ir_conf(40, 5, 120, 48, True, "HS", 1, 1),
                ir_conf(48, 5, 144, 48, True, "HS", 1, 1),
                ir_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
                ir_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
                ir_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            ]
        else:
            raise ValueError("Unsupported model type {}".format(arch))
        self.widen_factor = widen_factor
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        if norm_cfg is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
        else:
            norm_layer = get_norm(norm_cfg)

        self.in_channels = make_divisible(16 * widen_factor, 8)
        # conv1
        conv1_output_channels = inverted_residual_setting[0].input_channels
        # 1/2
        self.conv1 = ConvNormActivation(
            input_channels,
            conv1_output_channels,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=act_cfg if act_cfg else nn.Hardswish,
        )

        self.layers = []

        for i, conf in enumerate(inverted_residual_setting):
            layer = InvertedResidual(conf, norm_layer=norm_layer)
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

    def forward(self, x):
        res = []
        x = self.conv1(x)
        for i, layer_name in enumerate(self.layers):
            x = getattr(self, layer_name)(x)
            if i in self.out_indices:
                res.append(x)
                if i == max(self.out_indices):
                    break

        return res

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(MobileNetV3, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
