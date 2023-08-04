import copy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from mmcv.transforms.base import BaseTransform

from edgelab.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Bbox2FomoMask(BaseTransform):
    def __init__(
        self,
        downsample_factor: Tuple[int, ...] = (8,),
        num_classes: int = 80,
    ) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor
        self.num_classes = num_classes

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        results['img']
        H, W = results['img_shape']
        bbox = results['gt_bboxes']
        labels = results['gt_bboxes_labels']

        res = []
        for factor in self.downsample_factor:
            Dh, Dw = int(H / factor), int(W / factor)
            target = self.build_target(bbox, feature_shape=(Dh, Dw), ori_shape=(W, H), labels=labels)
            res.append(target)

        results['fomo_mask'] = copy.deepcopy(res)
        return results

    def build_target(self, bboxs, feature_shape, ori_shape, labels):
        (H, W) = feature_shape
        # target_data = torch.zeros(size=(1,H, W, self.num_classes + 1))
        target_data = np.zeros((1, H, W, self.num_classes + 1))
        target_data[..., 0] = 1
        for idx, i in enumerate(bboxs):
            w = int(i.centers[0][0] / ori_shape[0] * H)
            h = int(i.centers[0][1] / ori_shape[1] * W)
            target_data[0, h, w, 0] = 0  # background
            target_data[0, h, w, int(labels[idx] + 1)] = 1  # label
        return target_data
