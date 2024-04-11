# copyright Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcls.models.classifiers.base import BaseClassifier
from mmcls.structures import ClsDataSample

from sscma.registry import MODELS


@MODELS.register_module()
class AccelerometerClassifier(BaseClassifier):
    """Accelerometer classifiers for supervised classification task.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmcls.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmcls.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmcls.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:

            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in :mod:`mmcls.model.utils.augment`.

            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "SensorDataPreprocessor" as type. See :class:`SensorDataPreprocessor` for
            more details. Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(
        self,
        backbone: dict,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        pretrained: Optional[str] = None,
        train_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'sscma.SensorDataPreprocessor')

        if train_cfg is not None and 'augments' in train_cfg:
            # Set batch augmentations by `train_cfg`
            data_preprocessor['batch_augments'] = train_cfg

        super(AccelerometerClassifier, self).__init__(init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, inputs: torch.Tensor, data_samples: Optional[List[ClsDataSample]] = None, mode: str = 'tensor'):
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            head_out = self.head(feats) if self.with_head else feats
            if head_out.shape[-1] > 2:
                head_out = F.softmax(head_out, dim=-1)
            else:
                head_out = torch.sigmoid(head_out)
            return head_out
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')

    def extract_feat(self, inputs, stage='neck'):
        assert stage in ['backbone', 'neck', 'pre_logits'], (
            f'Invalid output stage "{stage}", please choose from "backbone", ' '"neck" and "pre_logits"'
        )

        x = self.backbone(inputs)

        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        assert self.with_head and hasattr(
            self.head, 'pre_logits'
        ), "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor, data_samples: List[ClsDataSample]) -> dict:
        feats = self.extract_feat(inputs)

        return self.head.loss(feats, data_samples)

    def predict(
        self, inputs: torch.Tensor, data_samples: Optional[List[ClsDataSample]] = None, **kwargs
    ) -> List[ClsDataSample]:
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples, **kwargs)
