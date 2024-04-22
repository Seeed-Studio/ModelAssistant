# copyright Copyright (c) Seeed Technology Co.,Ltd.
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import torch
from .base import BaseDetector
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmengine.dist import get_world_size

from torch import Tensor

from sscma.registry import MODELS


@MODELS.register_module()
class SingleStageDetector(BaseDetector):
    def __init__(
        self,
        backbone: OptConfigType,
        neck: OptConfigType = None,
        bbox_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(data_preprocessor, init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: Union[List[str], str],
        unexpected_keys: Union[List[str], str],
        error_msgs: Union[List[str], str],
    ) -> None:
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [k for k in state_dict.keys() if k.startswith(bbox_head_prefix)]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [k for k in state_dict.keys() if k.startswith(rpn_head_prefix)]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + rpn_head_key[len(rpn_head_prefix) :]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        losses = self.bbox_head.loss(x, batch_data_samples)
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x


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
