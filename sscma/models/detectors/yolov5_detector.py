# Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) OpenMMLab.
from typing import Union

import torch
from torch import Tensor
from mmengine.dist import get_world_size
from mmengine.registry import MODELS

from sscma.utils.typing_utils import ConfigType, OptConfigType, OptMultiConfig

from .single_stage import SingleStageDetector


@MODELS.register_module()
class YOLODetector(SingleStageDetector):
    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        use_syncbn: bool = True,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )

        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
