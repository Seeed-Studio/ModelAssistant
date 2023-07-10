from typing import Dict, List, Optional, Tuple, Union

import torch
from mmcv.transforms.base import BaseTransform

from edgelab.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Bbox2FomoMask(BaseTransform):
    def __init__(
        self,
        downsample_factor: Tuple[int, ...] = (8,),
        classes_num: int = 80,
    ) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor
        self.classes_num = classes_num

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        H, W = results['img_shape']
        bbox = results['gt_bboxes']
        print(bbox)

        res = []
        for factor in self.downsample_factor:
            Dh, Dw = H / factor, W / factor
            target = self.build_target(bbox, shape=(Dh, Dw))
            res.append(target)

        results['fomo_mask'] = res
        return results

    def build_target(self, targets, shape):
        (H, W) = shape
        target_data = torch.zeros(size=(H, W, self.classes_num + 1))
        target_data[..., 0] = 1
        for i in targets:
            h, w = int(i[3].item() * H), int(i[2].item() * W)
            target_data[int(i[0]), h, w, 0] = 0  # background
            target_data[int(i[0]), h, w, int(i[1])] = 1  # label

        return target_data
