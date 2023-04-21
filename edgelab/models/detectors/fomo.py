from typing import Optional, Dict
from mmdet.models.detectors.single_stage import SingleStageDetector
from edgelab.registry import MODELS


@MODELS.register_module()
class Fomo(SingleStageDetector):

    def __init__(self,
                 backbone: Dict,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 data_preprocessor: Optional[Dict] = None,
                 train_cfg: Optional[Dict] = None,
                 test_cfg: Optional[Dict] = None,
                 pretrained: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None):
        super().__init__(backbone, neck, head, train_cfg, test_cfg,
                         data_preprocessor, init_cfg)
