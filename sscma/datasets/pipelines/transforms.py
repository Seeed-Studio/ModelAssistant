import copy
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from mmcv.transforms.base import BaseTransform

from sscma.registry import TRANSFORMS


@TRANSFORMS.register_module()
class Color2Gray(BaseTransform):
    def __init__(self, one_channel: bool = False, conver_order: Optional[str] = None) -> None:
        super().__init__()
        if one_channel and conver_order is not None:
            raise ValueError("one_channel and conver_order can only set one of them, not all of them ")

        if conver_order is not None:
            if not hasattr(cv2, "COLOR_" + conver_order):
                opt = ','.join(
                    (map(lambda x: x.replace("COLOR_", ""), filter(lambda x: x.startswith("COLOR_"), dir(cv2))))
                )
                raise ValueError(
                    f"The value of convert_order can only be one of the following[{opt}], but {conver_order} is obtained"
                )
            self.conver_opt = getattr(cv2, "COLOR_" + conver_order)
        self.one_channel = one_channel

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        img = results['img']
        if self.one_channel:
            img = img[..., 0:1]
        else:
            img = np.expand_dims(cv2.cvtColor(img, self.conver_opt, dstCn=1), -1)
        results['img'] = img
        return results


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
