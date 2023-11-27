import copy
from typing import Tuple
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model.base_module import BaseModule
from mmpose.models.backbones.shufflenet_v2 import ShuffleNetV2

from sscma.registry import BACKBONES


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

        stage_names = ["stage2", "stage3", "stage4"]
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
