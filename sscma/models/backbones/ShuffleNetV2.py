# Copyright (c) Seeed Tech Ltd. All rights reserved.
import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn
from mmdet.models.utils.make_divisible import make_divisible
from mmengine.model import BaseModule
from torch import Tensor

from sscma.models.base import ConvModule
from sscma.models.base.general import ConvNormActivation
from sscma.registry import BACKBONES


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
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Identity()

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)
        return out


@BACKBONES.register_module(name='CusShuffleNetV2')
class ShuffleNetV2(BaseModule):
    def __init__(
        self,
        widen_factor=1,
        out_indices=(2,),
        frozen_stages=-1,
        input_channels: int = 3,
        conv_cfg=dict(type='Conv'),
        norm_cfg=None,
        act_cfg=dict(type='ReLu'),
        norm_eval=False,
        reduced_tail=False,
        dilate=False,
        pretrained=None,
        init_cfg: Optional[dict] = None,
    ):
        super(ShuffleNetV2, self).__init__(init_cfg)
        arch = {
            '0.25': [24, 24, 48, 96, 512],
            '0.5': [24, 48, 96, 192, 1024],
            '1.0': [24, 116, 232, 464, 1024],
            '1.5': [24, 176, 352, 704, 1024],
            '2.0': [24, 244, 488, 976, 2048],
        }

        layer_repeats = [4, 8, 4]
        if str(widen_factor) not in arch.keys():
            tmp_channel = arch['1.0']
            perchannel_widen = []
            for i in tmp_channel:
                perchannel_widen.append(make_divisible(i * float(widen_factor), 8))
        else:
            perchannel_widen = arch[str(widen_factor)]

        output_channels = perchannel_widen[0]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.conv1 = ConvNormActivation(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            norm_layer='BatchNorm2d',
            activation_layer='ReLU',
        )

        input_channels = output_channels
        self.layer_names = [f'layer{i}' for i in [1, 2, 3]]
        for name, repeats, output_channels in zip(self.layer_names, layer_repeats, perchannel_widen[1:]):
            layer = [InvertedResidual(input_channels, output_channels, 2)]
            for _ in range(repeats - 1):
                layer.append(InvertedResidual(output_channels, output_channels, 1))
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

    def _make_layer(self, out_channels, num_blocks):
        """Stack blocks to make a layer.

        Args:
            out_channels (int): out_channels of the block.
            num_blocks (int): number of blocks.
        """
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            layers.append(
                InvertedResidual(
                    in_channels=self.in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp,
                )
            )
            self.in_channels = out_channels

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


class ShuffleV2Block(nn.Module):
    """
    Reference: https://github.com/dog-qiuqiu/FastestDet/blob/50473cd155cb088aa4a99e64ff6a4b3c24fa07e1/module/shufflenetv2.py#L4
    """

    def __init__(self, inp, oup, mid_channels, *, ksize, stride) -> None:
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x) -> torch.Tensor:
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x) -> Tuple[torch.Tensor]:
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % 4 == 0
        x = x.reshape(batchsize * num_channels // 2, 2, height * width)
        x = x.permute(1, 0, 2)
        x = x.reshape(2, -1, num_channels // 2, height, width)
        return x[0], x[1]


@BACKBONES.register_module()
class FastShuffleNetV2(BaseModule):
    """
    Reference: https://github.com/dog-qiuqiu/FastestDet/blob/50473cd155cb088aa4a99e64ff6a4b3c24fa07e1/module/shufflenetv2.py#L64C6-L64C7
    """

    def __init__(self, stage_repeats, stage_out_channels, *args, **kwargs) -> None:
        super(FastShuffleNetV2, self).__init__(*args, **kwargs)

        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage2', 'stage3', 'stage4']
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(
                        ShuffleV2Block(
                            input_channel, output_channel, mid_channels=output_channel // 2, ksize=3, stride=2
                        )
                    )
                else:
                    stageSeq.append(
                        ShuffleV2Block(
                            input_channel // 2, output_channel, mid_channels=output_channel // 2, ksize=3, stride=1
                        )
                    )
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))

    def forward(self, x) -> Tuple[torch.Tensor]:
        x = self.first_conv(x)
        x = self.maxpool(x)
        P1 = self.stage2(x)
        P2 = self.stage3(P1)
        P3 = self.stage4(P2)

        return (P1, P2, P3)


@BACKBONES.register_module()
class CustomShuffleNetV2(ShuffleNetV2):
    def __init__(
        self,
        widen_factor=1.0,
        out_indices=(3,),
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        norm_eval=False,
        with_cp=False,
    ):
        # super().__init__(widen_factor, out_indices, frozen_stages, conv_cfg,
        #                  norm_cfg, act_cfg, norm_eval, with_cp)

        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.stage_blocks = [4, 8, 4]
        for index in out_indices:
            if index not in range(0, 4):
                raise ValueError('the item in out_indices must in ' f'range(0, 4). But received {index}')

        if frozen_stages not in range(-1, 4):
            raise ValueError('frozen_stages must be in range(-1, 4). ' f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        if widen_factor == 0.25:
            channels = [48, 96, 192]
        elif widen_factor == 0.5:
            channels = [48, 96, 192, 1024]
        elif widen_factor == 1.0:
            channels = [116, 232, 464, 1024]
        elif widen_factor == 1.5:
            channels = [176, 352, 704, 1024]
        elif widen_factor == 2.0:
            channels = [244, 488, 976, 2048]
        else:
            raise ValueError('widen_factor must be in [0.25, 0.5, 1.0, 1.5, 2.0]. ' f'But received {widen_factor}')

        self.in_channels = 24
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = nn.ModuleList()
        for i, num_blocks in enumerate(self.stage_blocks):
            layer = self._make_layer(channels[i], num_blocks)
            self.layers.append(layer)

        output_channels = channels[-1]
        self.layers.append(
            ConvModule(
                in_channels=self.in_channels,
                out_channels=output_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            )
        )
