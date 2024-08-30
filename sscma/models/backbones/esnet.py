# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

from mmengine.model import BaseModule

import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..cnn import ConvModule, EnhancedInvertedResidual, EnhancedInvertedResidualDS, make_divisible


class ESNet(BaseModule):
    """Enhanced ShuffleNet used in PicoDet

    Args:
        model_size (str) Model size of ESNet.
        out_indicts: (Sequence[int]): Output from which stages.
            Default: (4, 11, 14).
        frozen_stages (int): Stages to be frozen (stop grad and set eval
            mode). -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        se_cfg (dict): Config dict for squeeze and excitation layer.
            Default: dict(conv_cfg=None, ratio=4,
                          act_cfg=(dict(type='ReLU'), dict(type='HSigmoid'))).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    # Parameters to build layers
    arch_settings = {
        's': {'scale': 0.75,
            'ratio': [0.875, 0.5, 0.5, 0.5, 0.625, 0.5, 0.625,
                      0.5, 0.5, 0.5, 0.5, 0.5, 0.5]},
        'm': {'scale': 1.0,
            'ratio': [0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625,
                      0.625, 0.5, 0.625, 1.0, 0.625, 0.75]},
        'l': {'scale': 1.25,
            'ratio': [0.875, 0.5, 1.0, 0.625, 0.5, 0.75, 0.625,
                    0.625, 0.5, 0.625, 1.0, 0.625, 0.75]},
    }
    stage_repeats = [3, 7, 3]

    def __init__(
        self,
        model_size="m",
        out_indices=(2, 9, 12),
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type="BN", requires_grad=True),
        act_cfg=dict(type="HardSwish"),
        norm_eval=False,
        se_cfg=dict(
            conv_cfg=None, ratio=4, act_cfg=(dict(type="ReLU"), dict(type="HSigmoid"))
        ),
        with_cp=False,
        pretrained=None,
        init_cfg=None,
    ):
        super(ESNet, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (
            init_cfg and pretrained
        ), "init_cfg and pretrained cannot be specified at the same time"
        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type="Kaiming", layer="Conv2d"),
                    dict(type="Constant", val=1, layer=["_BatchNorm", "GroupNorm"]),
                ]
        else:
            raise TypeError("pretrained must be a str or None")

        self.model_size = model_size
        assert model_size in [
            "s",
            "m",
            "l",
        ], "invalid model_size {}, \
                select model size of {}".format(
            model_size, list(self.arch_settings.keys())
        )
        self.out_indices = out_indices
        if not set(out_indices).issubset(set(range(1, 15))):
            raise ValueError(
                "out_indices must be a subset of range"
                f"[1, 15). But received {out_indices}"
            )
        if frozen_stages not in range(-1, 4):
            raise ValueError(
                "frozen_stages must be in range(-1, 4). "
                f"But received {frozen_stages}"
            )
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.se_cfg = se_cfg
        self.with_cp = with_cp

        self.scale = self.arch_settings[model_size]["scale"]
        self.channel_ratio = self.arch_settings[model_size]["ratio"]

        stage_out_channels = [
            -1,
            24,
            make_divisible(128 * self.scale, divisor=16),
            make_divisible(256 * self.scale, divisor=16),
            make_divisible(512 * self.scale, divisor=16),
            1024,
        ]

        self._out_channels = []
        self._feature_idx = 0
        # 1. conv1
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=stage_out_channels[1],
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 2. bottleneck sequences
        self.block_list = []
        arch_idx = 0
        for stage_id, num_repeat in enumerate(self.stage_repeats):
            for i in range(num_repeat):
                channels_scales = self.channel_ratio[arch_idx]
                mid_c = make_divisible(
                    int(stage_out_channels[stage_id + 2] * channels_scales), divisor=8
                )
                if i == 0:
                    self.se_cfg["channels"] = mid_c // 2
                    block = EnhancedInvertedResidualDS(
                        in_channels=stage_out_channels[stage_id + 1],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=2,
                        se_cfg=self.se_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        with_cp=self.with_cp,
                        init_cfg=self.init_cfg,
                    )
                else:
                    self.se_cfg["channels"] = mid_c
                    block = EnhancedInvertedResidual(
                        in_channels=stage_out_channels[stage_id + 2],
                        mid_channels=mid_c,
                        out_channels=stage_out_channels[stage_id + 2],
                        stride=1,
                        se_cfg=self.se_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        with_cp=self.with_cp,
                        init_cfg=self.init_cfg,
                    )

                name = str(stage_id + 2) + "_" + str(i + 1)
                setattr(self, name, block)
                self.block_list.append(block)
                arch_idx += 1

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool(out)
        outs = []
        for i, block in enumerate(self.block_list):
            out = block(out)
            if i in self.out_indices:
                outs.append(out)
        return outs

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):

            block_num = self.stage_repeats[i]
            for num in range(block_num):
                layer = getattr(self, f"{i + 1}_{num + 1}")
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(ESNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
