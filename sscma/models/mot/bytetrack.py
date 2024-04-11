# copyright Copyright (c) Seeed Technology Co.,Ltd.
from mmdet.models import build_detector
from mmdet.models.detectors.base import BaseDetector
from mmtrack.models.builder import MODELS
from mmtrack.models.mot.base import BaseMultiObjectTracker


@MODELS.register_module()
class ByteTrack(BaseMultiObjectTracker):
    def __init__(self, detector=None, motion=None, tracker=None, init_cfg=None):
        super().__init__(init_cfg)

        if detector is not None:
            self.detector: BaseDetector = build_detector(detector)

        if motion is not None:
            self.motion = motion

        if tracker is not None:
            self.tracker = tracker

    def forward_train(self, imgs, img_metas, **kwargs):
        return self.detector.forward_train(imgs, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        pass
