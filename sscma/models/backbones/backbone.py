# Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) OpenMMLab.
from typing import List, Tuple, Union

import torch

from sscma.utils import ConfigType, OptMultiConfig
from ..cnn import ConvModule
from ..layers import CSPLayer
from ..necks import SPPFBottleneck
from ..utils import make_divisible, make_round
from .base_backbone import YOLOBaseBackbone


class YOLOv5CSPDarknet(YOLOBaseBackbone):
    arch_settings = {
        "P5": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 9, True, False],
            [512, 1024, 3, True, True],
        ],
        "P6": [
            [64, 128, 3, True, False],
            [128, 256, 6, True, False],
            [256, 512, 9, True, False],
            [512, 768, 3, True, False],
            [768, 1024, 3, True, True],
        ],
    }

    def __init__(
        self,
        arch: str = "P5",
        plugins: Union[dict, List[dict]] = None,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Tuple[int] = (2, 3, 4),
        frozen_stages: int = -1,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        norm_eval: bool = False,
        init_cfg: OptMultiConfig = None,
    ):
        self.arch = arch
        super().__init__(
            self.arch_settings[arch],
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg,
        )

    def build_stem_layer(self):
        return ConvModule(
            self.input_channels,
            make_divisible(self.arch_settings[self.arch][0][0], self.widen_factor),
            kernel_size=6,
            stride=2,
            padding=2,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def build_stage_layer(self, stage_idx: int, setting: list) -> list:
        in_channels, out_channels, num_blocks, add_identity, use_spp = setting

        in_channels = make_divisible(in_channels, self.widen_factor)
        out_channels = make_divisible(out_channels, self.widen_factor)
        num_blocks = make_round(num_blocks)
        stage = []
        conv_layer = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        stage.append(conv_layer)

        csp_layer = CSPLayer(
            out_channels,
            out_channels,
            num_blocks=num_blocks,
            add_identity=add_identity,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        stage.append(csp_layer)

        if use_spp:
            spp = SPPFBottleneck(
                out_channels,
                out_channels,
                kernel_sizes=5,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            )
            stage.append(spp)
        return stage

    def init_weights(self):
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    m.reset_parameters()
        else:
            super().init_weights()
