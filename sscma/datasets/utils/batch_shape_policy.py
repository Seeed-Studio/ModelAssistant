# copyright Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) Seeed Tech Ltd.
# Copyright (c) OpenMMLab.
from typing import List

import numpy as np

from sscma.registry import TASK_UTILS


@TASK_UTILS.register_module()
class BatchShapePolicy:
    def __init__(self, batch_size: int = 32, img_size: int = 640, size_divisor: int = 32, extra_pad_ratio: float = 0.5):
        self.batch_size = batch_size
        self.img_size = img_size
        self.size_divisor = size_divisor
        self.extra_pad_ratio = extra_pad_ratio

    def __call__(self, data_list: List[dict]) -> List[dict]:
        image_shapes = []
        for data_info in data_list:
            image_shapes.append((data_info['width'], data_info['height']))

        image_shapes = np.array(image_shapes, dtype=np.float64)

        n = len(image_shapes)  # number of images
        batch_index = np.floor(np.arange(n) / self.batch_size).astype(np.int64)  # batch index
        number_of_batches = batch_index[-1] + 1  # number of batches

        aspect_ratio = image_shapes[:, 1] / image_shapes[:, 0]  # aspect ratio
        irect = aspect_ratio.argsort()

        data_list = [data_list[i] for i in irect]

        aspect_ratio = aspect_ratio[irect]
        # Set training image shapes
        shapes = [[1, 1]] * number_of_batches
        for i in range(number_of_batches):
            aspect_ratio_index = aspect_ratio[batch_index == i]
            min_index, max_index = aspect_ratio_index.min(), aspect_ratio_index.max()
            if max_index < 1:
                shapes[i] = [max_index, 1]
            elif min_index > 1:
                shapes[i] = [1, 1 / min_index]

        batch_shapes = (
            np.ceil(np.array(shapes) * self.img_size / self.size_divisor + self.extra_pad_ratio).astype(np.int64)
            * self.size_divisor
        )

        for i, data_info in enumerate(data_list):
            data_info['batch_shape'] = batch_shapes[batch_index[i]]

        return data_list
