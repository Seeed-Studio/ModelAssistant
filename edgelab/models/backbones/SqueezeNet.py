import torch
from mmcv.runner.base_module import BaseModule
import torch.nn as nn
from typing import Optional
from edgelab.models.base.general import ConvNormActivation
from mmdet.models.builder import BACKBONES
from mmdet.models.utils.make_divisible import make_divisible


class Squeeze(nn.Module):
    
    def __init__(self, inplanes: int, squeeze_planes: int,
                 expand_planes: int) -> None:
        super(Squeeze, self).__init__()
        self.inplanes = inplanes
        expand1x1_planes = expand_planes // 2
        expand3x3_planes = expand_planes // 2
        self.squeeze = ConvNormActivation(inplanes,
                                          squeeze_planes,
                                          kernel_size=1,
                                          activation_layer='ReLU')

        self.expand1x1 = ConvNormActivation(squeeze_planes,
                                            expand1x1_planes,
                                            kernel_size=1,
                                            activation_layer='ReLU')

        self.expand3x3 = ConvNormActivation(squeeze_planes,
                                            expand3x3_planes,
                                            kernel_size=3,
                                            padding=1,
                                            activation_layer='ReLU')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], 1)


@BACKBONES.register_module()
class SqueezeNet(BaseModule):
    arch = [[128, 16, 128], [128, 32, 256], [256, 32, 256], [256, 48, 384],
            [384, 48, 384], [384, 64, 512], [512, 64, 512]]

    def __init__(self,
                 input_channels: int = 3,
                 widen_factor: float = 1.0,
                 out_indices=(1, ),
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        arch_setting=self.arch
        if widen_factor == 1.0:
            frist_conv = (input_channels, 96, 7, 2, 0)
            frist_setting = [96, 16, 128]
            arch_setting.insert(0, frist_setting)
        elif widen_factor == 1.1:
            frist_conv = (input_channels, 64, 3, 2, 0)
            frist_setting = [64, 16, 128]
            arch_setting.insert(0, frist_setting)
        else:
            frist_out = make_divisible(64 * widen_factor, 8)
            frist_conv = (input_channels, frist_out, 3, 2, 1)
            frist_setting = [64, 16, 128]
            arch_setting.insert(0, frist_setting)
            for i, setting in enumerate(arch_setting):
                arch_setting[i] = [
                    make_divisible(setting[0] * widen_factor, 8) if i!=0 else frist_out, setting[1],
                    make_divisible(setting[-1] * widen_factor, 8)
                ]

        self.out_indices = out_indices
        self.frozen_stages=frozen_stages
        self.norm_eval=norm_eval
        self.max_pool_index=[1,3]

        self.conv1 = ConvNormActivation(*frist_conv, activation_layer='ReLU')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.layer_name = [
            f'layer{i}' for i in range(1,
                                       len(arch_setting) + 1)
        ]
        for name, param in zip(self.layer_name, arch_setting):

            layer = Squeeze(*param)
            
            self.add_module(name, layer)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        res = []
        for i, name in enumerate(self.layer_name):
            x = getattr(self, name)(x)
            if i in self.max_pool_index:
                x=self.maxpool(x)
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
        super(SqueezeNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
