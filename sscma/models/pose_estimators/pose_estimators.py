# copyright Copyright (c) Seeed Technology Co.,Ltd.
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple, Union

import torch
from mmengine.config import ConfigDict
from mmengine.model import BaseModel
from torch import Tensor

from sscma.registry import MODELS
from sscma.structures import PoseDataSample

from .utils import get_world_size, parse_pose_metainfo


class BasePoseEstimator(BaseModel, metaclass=ABCMeta):
    def __init__(
        self,
        backbone: Union[ConfigDict, dict],
        neck: Optional[Union[ConfigDict, dict]] = None,
        head: Optional[Union[ConfigDict, dict]] = None,
        train_cfg: Optional[Union[ConfigDict, dict]] = None,
        test_cfg: Optional[Union[ConfigDict, dict]] = None,
        data_preprocessor: Optional[Union[ConfigDict, dict]] = None,
        use_syncbn: bool = False,
        init_cfg: Optional[Union[Union[ConfigDict, dict], List[Union[ConfigDict, dict]]]] = None,
        metainfo: Optional[dict] = None,
    ):
        super().__init__(data_preprocessor, init_cfg)
        self.metainfo = self._load_metainfo(metainfo=metainfo)
        self.train_cfg = train_cfg if train_cfg else {}
        self.test_cfg = test_cfg if test_cfg else {}

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)
            self.head.test_cfg = self.test_cfg.copy()

        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None

    def forward(self, inputs: torch.Tensor, data_samples: List[PoseDataSample], mode: str = 'tensor'):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs)
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            # use customed metainfo to override the default metainfo
            if self.metainfo is not None:
                for data_sample in data_samples:
                    data_sample.set_metainfo(self.metainfo)
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". ' 'Only supports loss, predict and tensor mode.')

    @abstractmethod
    def loss(self, inputs: Tensor, data_samples: List[PoseDataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

    @abstractmethod
    def predict(self, inputs: Tensor, data_samples: List[PoseDataSample]) -> List[PoseDataSample]:
        """Predict results from a batch of inputs and data samples with post-
        processing."""

    def _forward(self, inputs: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        x = self.backbone(inputs)

        if self.with_neck:
            x = self.neck(x)

        if self.with_head:
            x = self.head.forward(x)

        return x

    def _load_metainfo(self, metainfo: dict = None) -> dict:
        if metainfo is None:
            return None

        if not isinstance(metainfo, dict):
            raise TypeError(f'metainfo should be a dict, but got {type(metainfo)}')

        metainfo = parse_pose_metainfo(metainfo=metainfo)
        return metainfo

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)

        return x
