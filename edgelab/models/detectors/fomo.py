import torch
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

    def _forward(self, batch_inputs, batch_data_samples):
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return [
            torch.softmax(pred, dim=1).permute(0, 2, 3, 1) for pred in results
        ]
