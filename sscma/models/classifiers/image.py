from typing import List, Optional

import torch
import torch.nn.functional as F
from mmcls.models.classifiers import ImageClassifier as MMImageClassifier
from mmcls.structures import ClsDataSample

from sscma.registry import MODELS


@MODELS.register_module()
class ImageClassifier(MMImageClassifier):
    def __init__(
        self,
        backbone: dict,
        neck: Optional[dict] = None,
        head: Optional[dict] = None,
        pretrained: Optional[str] = None,
        train_cfg: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        softmax: bool = True,
        init_cfg: Optional[dict] = None,
    ):
        super(ImageClassifier, self).__init__(backbone, neck, head, pretrained, train_cfg, data_preprocessor, init_cfg)

        self.softmax = softmax

    def forward(self, inputs: torch.Tensor, data_samples: Optional[List[ClsDataSample]] = None, mode: str = 'tensor'):
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            head_out = self.head(feats) if self.with_head else feats
            return F.softmax(head_out, dim=1) if self.softmax else head_out
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
