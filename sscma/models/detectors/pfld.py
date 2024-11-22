# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from typing import Optional, Tuple

import numpy as np
from torch import Tensor

from mmengine import MODELS
from mmengine.structures.instance_data import InstanceData
from mmengine.model import BaseModel
from sscma.structures import PoseDataSample


class PFLD(BaseModel):
    """
    PFLD: A Practical Facial Landmark Detector: https://arxiv.org/abs/1902.10859
    Args:
        backbone(dict): Configuration of pfld model backbone
        head(dict): Configuration of pfld model head
        pretrained: Model pre-training weight path
    """

    def __init__(self, backbone: dict, head: dict, pretrained: Optional[str] = None):
        super(PFLD, self).__init__()
        self.backbone = MODELS.build(backbone)
        self.head = MODELS.build(head)
        self.pretrained = pretrained

    def forward(self, inputs, data_samples=None, mode="tensor"):
        if mode == "loss":
            return self.loss(inputs, data_samples)
        elif mode == "predict":
            return self.predict(inputs, data_samples)
        elif mode == "tensor":
            return self.forward_(inputs, data_samples)
        else:
            raise ValueError(f"params mode receive a not exception params:{mode}")

    def loss(self, inputs, data_samples):
        x = self.extract_feat(inputs)
        results = dict()
        results.update(self.head.loss(x, data_samples))
        return results

    def predict(self, inputs, data_samples):
        feat = self.extract_feat(inputs)
        x = self.head.predict(feat)
        res = PoseDataSample(**data_samples)
        res.results = x
        res.pred_instances = InstanceData(
            keypoints=np.array([x.reshape(-1, 2).cpu().numpy()])
            * data_samples["init_size"][1].reshape(-1, 1).cpu().numpy()
        )

        return [res]

    def forward_(self, inputs, data_samples):
        x = self.extract_feat(inputs)
        return self.head(x)

    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        x = self.backbone(inputs)
        if self.with_neck:
            x = self.neck(x)

        return x

    @property
    def with_neck(self) -> bool:
        return hasattr(self, "neck") and self.neck is not None
