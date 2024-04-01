from typing import Optional

import numpy as np
from mmengine.structures.instance_data import InstanceData

from sscma.models.pose_estimators import BasePoseEstimator
from sscma.registry import MODELS
from sscma.structures import PoseDataSample


@MODELS.register_module()
class PFLD(BasePoseEstimator):
    """
    PFLD: A Practical Facial Landmark Detector: https://arxiv.org/abs/1902.10859
    Args:
        backbone(dict): Configuration of pfld model backbone
        head(dict): Configuration of pfld model head
        pretrained: Model pre-training weight path
    """

    def __init__(self, backbone: dict, head: dict, pretrained: Optional[str] = None):
        super(PFLD, self).__init__(backbone, head=head)
        self.backbone = MODELS.build(backbone)
        self.head = MODELS.build(head)
        self.pretrained = pretrained

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self.forward_(inputs, data_samples)
        else:
            raise ValueError(f'params mode receive a not exception params:{mode}')

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
            keypoints=np.array([x.reshape(-1, 2).cpu().numpy()]) * data_samples['init_size'][1].cpu().numpy()
        )

        return [res]

    def forward_(self, inputs, data_samples):
        x = self.extract_feat(inputs)
        return self.head(x)
