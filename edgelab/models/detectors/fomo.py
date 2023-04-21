from mmdet.models.detectors.single_stage import SingleStageDetector
from edgelab.registry import MODELS


@MODELS.register_module()
class Fomo(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 data_preprocessor=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None
                 ):
        super().__init__(backbone, neck, head, train_cfg, test_cfg, 
                         data_preprocessor,init_cfg)

    def predict(self, imgs, data_samples):
        x = self.extract_feat(imgs)
        return self.bbox_head.predict(x, data_samples)
