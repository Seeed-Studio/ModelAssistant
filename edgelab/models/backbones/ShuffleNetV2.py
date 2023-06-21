import torchvision
import torch
from torch import Tensor
from typing import Optional
import torch.nn as nn
from mmdet.models.utils.make_divisible import make_divisible
from mmengine.model import BaseModule

from edgelab.registry import BACKBONES
from edgelab.models.base.general import ConvNormActivation


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp: int, oup: int, stride: int) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp,
                                    inp,
                                    kernel_size=3,
                                    stride=self.stride,
                                    padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp,
                          branch_features,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features,
                                branch_features,
                                kernel_size=3,
                                stride=self.stride,
                                padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i: int,
                       o: int,
                       kernel_size: int,
                       stride: int = 1,
                       padding: int = 0,
                       bias: bool = False) -> nn.Conv2d:
        return nn.Conv2d(i,
                         o,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias,
                         groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


@BACKBONES.register_module(name='TmpShuffleNetV2')
class ShuffleNetV2(BaseModule):

    def __init__(self,
                 widen_factor=1,
                 out_indices=(2, ),
                 frozen_stages=-1,
                 input_channels: int = 3,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLu'),
                 norm_eval=False,
                 reduced_tail=False,
                 dilate=False,
                 pretrained=None,
                 init_cfg: Optional[dict] = None):
        super(ShuffleNetV2, self).__init__(init_cfg)
        arch = {
            "0.25": [24, 24, 48, 96, 512],
            "0.5": [24, 48, 96, 192, 1024],
            "1.0": [24, 116, 232, 464, 1024],
            "1.5": [24, 176, 352, 704, 1024],
            "2.0": [24, 244, 488, 976, 2048]
        }

        layer_repeats = [4, 8, 4]
        if str(widen_factor) not in arch.keys():
            tmp_channel = arch['1.0']
            perchannel_widen = []
            for i in tmp_channel:
                perchannel_widen.append(
                    make_divisible(i * float(widen_factor), 8))
        else:
            perchannel_widen = arch[str(widen_factor)]

        output_channels = perchannel_widen[0]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.conv1 = ConvNormActivation(input_channels,
                                        output_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                        norm_layer='BatchNorm2d',
                                        activation_layer='ReLU')

        input_channels = output_channels
        self.layer_names = [f'layer{i}' for i in [1, 2, 3]]
        for name, repeats, output_channels in zip(self.layer_names,
                                                  layer_repeats,
                                                  perchannel_widen[1:]):
            layer = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                layer.append(
                    InvertedResidual(output_channels, output_channels, 1))
            input_channels = output_channels
            self.add_module(name, nn.Sequential(*layer))

    def mask_layer(self):
        pass

    def forward(self, x):
        x = self.conv1(x)
        res = []

        for i, name in enumerate(self.layer_names):
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
        super(ShuffleNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
