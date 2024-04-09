# Copyright (c) Seeed Tech Ltd. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Sequence

from mmdet.utils import ConfigType
from mmengine.model import BaseModule

from sscma.registry import MODELS


@MODELS.register_module()
class BaseBackbone(BaseModule, metaclass=ABCMeta):
    def __init__(
        self,
        arch_setting: list,
        deepen_factor: float = 1.0,
        widen_factor: float = 1.0,
        input_channels: int = 3,
        out_indices: Sequence[int] = (2, 3, 4),
        frozen_stages: int = -1,
        norm_cfg: ConfigType = None,
        act_cfg: ConfigType = None,
        norm_eval: bool = False,
        init_cfg: dict | List[dict] | None = None,
    ):
        super().__init__(init_cfg)
        self.num_stages = len(arch_setting)
        self.arch_setting = arch_setting

        if frozen_stages not in range(-1, len(arch_setting) + 1):
            raise ValueError('fronzen_stages value is invalid')

        self.input_channels = input_channels
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.widen_factor = widen_factor
        self.deepen_factor = deepen_factor
        self.norm_eval = norm_eval
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.stem = self.build_stem_layer()

    @abstractmethod
    def build_stem_layer(self):
        """Build a stem layer."""
        pass

    @abstractmethod
    def build_stage_layer(self):
        """Build a stage layer."""
        pass
