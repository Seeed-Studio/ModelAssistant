from typing import Optional, List, Callable
from functools import partial
import copy
import math
from torch import nn
import torch
from torchvision.ops import StochasticDepth
from mmdet.models.builder import BACKBONES
from mmdet.models.utils.make_divisible import make_divisible
from mmcv.runner.base_module import BaseModule
from edgelab.models.base.general import ConvNormActivation, SqueezeExcitation


class MBConvConfig:
    # Stores information listed at Table 1 of the EfficientNet paper
    def __init__(self, expand_ratio: float, kernel: int, stride: int,
                 input_channels: int, out_channels: int, num_layers: int,
                 width_mult: float, depth_mult: float) -> None:
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.num_layers = self.adjust_depth(num_layers, depth_mult)

    @staticmethod
    def adjust_channels(channels: int,
                        width_mult: float,
                        min_value: Optional[int] = None) -> int:
        return make_divisible(channels * width_mult, 8, min_value)

    @staticmethod
    def adjust_depth(num_layers: int, depth_mult: float):
        return make_divisible(num_layers * depth_mult, 1)


class MBConv(nn.Module):

    def __init__(
            self,
            cnf: MBConvConfig,
            stochastic_depth_prob: float,
            norm_layer: Callable[..., nn.Module],
            se_layer: Callable[..., nn.Module] = SqueezeExcitation) -> None:
        super().__init__()

        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels,
                                                cnf.expand_ratio)
        if expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(cnf.input_channels,
                                   expanded_channels,
                                   kernel_size=1,
                                   norm_layer=norm_layer,
                                   activation_layer=activation_layer))

        # depthwise
        layers.append(
            ConvNormActivation(expanded_channels,
                               expanded_channels,
                               kernel_size=cnf.kernel,
                               stride=cnf.stride,
                               groups=expanded_channels,
                               norm_layer=norm_layer,
                               activation_layer=activation_layer))

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(expanded_channels,
                     squeeze_channels,
                     activation=partial(nn.SiLU, inplace=True)))

        # project
        layers.append(
            ConvNormActivation(expanded_channels,
                               cnf.out_channels,
                               kernel_size=1,
                               norm_layer=norm_layer,
                               activation_layer=None))

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = cnf.out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


@BACKBONES.register_module(name='edgeEfficienNet', force=True)
class EfficientNet(BaseModule):
    # https://arxiv.org/pdf/1610.02357
    arch = [
        [1, 3, 1, 32, 16, 1],
        [6, 3, 2, 16, 24, 2],
        [6, 5, 2, 24, 40, 2],
        [6, 3, 2, 40, 80, 3],
        [6, 5, 1, 80, 112, 3],
        [6, 5, 2, 112, 192, 4],
        [6, 3, 1, 192, 320, 1],
    ]

    width_depth_mult = {
        'b0': [1.0, 1.0, 0.2],
        'b1': [1.0, 1.1, 0.2],
        'b2': [1.1, 1.2, 0.3],
        'b3': [1.2, 1.4, 0.3],
        'b4': [1.4, 1.8, 0.4],
        'b5': [1.6, 2.2, 0.5],
        'b6': [1.8, 2.6, 0.5],
        'b7': [2.0, 3.1, 0.5]
    }

    def __init__(self,
                 arch='b0',
                 input_channels=3,
                 out_indices=(2, ),
                 norm_cfg='BN',
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        assert arch in self.width_depth_mult.keys()

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.out_indices = out_indices

        width_depth_setting = self.width_depth_mult[arch]

        self.layer_name = [f'layer{i}' for i in range(1, len(self.arch) + 1)]
        block_conf = partial(MBConvConfig,
                             width_mult=width_depth_setting[0],
                             depth_mult=width_depth_setting[1])

        stochastic_depth_prob = width_depth_setting[-1]

        arch_param = [block_conf(*i) for i in self.arch]

        self.conv1 = ConvNormActivation(input_channels,
                                        arch_param[0].input_channels,
                                        3,
                                        2,
                                        norm_layer=norm_cfg,
                                        activation_layer='SiLU')

        total_stage_blocks = sum([cnf.num_layers for cnf in arch_param])
        stage_block_id = 0
        for name, param in zip(self.layer_name, arch_param):
            layer = []
            for _ in range(param.num_layers):
                conf = copy.copy(param)

                if layer:
                    conf.input_channels = conf.out_channels
                    conf.stride = 1
                sd_prob = stochastic_depth_prob * float(
                    stage_block_id) / total_stage_blocks
                layer.append(MBConv(conf, sd_prob, norm_layer=norm_cfg))
                stage_block_id += 1

            self.add_module(name, nn.Sequential(*layer))

    def forward(self, x):
        x = self.conv1(x)
        res = []
        for i, name in enumerate(self.layer_name):
            x = getattr(self, name)(x)
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
        super(EfficientNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
