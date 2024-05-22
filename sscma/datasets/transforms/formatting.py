# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from collections.abc import Sequence

import numpy as np
import torch
from sscma.structures import ClsDataSample
from mmcv.transforms import BaseTransform
from mmengine.utils import is_str

from sscma.registry import TRANSFORMS


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`'
        )


@TRANSFORMS.register_module()
class PackSensorInputs(BaseTransform):
    def __init__(self, meta_keys={'sample_idx', 'file_path', 'sensors'}):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Pack sensor inputs into a single tensor."""
        packed_results = dict()
        if 'data' in results:
            data = results['data']
            packed_results['inputs'] = to_tensor(data).to(dtype=torch.float32)

        data_sample = ClsDataSample()

        if 'gt_label' in results:
            gt_label = results['gt_label']
            data_sample.set_gt_label(gt_label)

        if self.meta_keys is not None:
            data_meta = {k: results[k] for k in self.meta_keys if k in results}
            data_sample.set_metainfo(data_meta)

        packed_results['data_samples'] = data_sample

        return packed_results


@TRANSFORMS.register_module()
class PackClsInputs(BaseTransform):
    """Pack the inputs data for the classification.

    **Required Keys:**

    - img
    - gt_label (optional)
    - ``*meta_keys`` (optional)

    **Deleted Keys:**

    All keys in the dict.

    **Added Keys:**

    - inputs (:obj:`torch.Tensor`): The forward data of models.
    - data_samples (:obj:`~sscma.structures.ClsDataSample`): The annotation
      info of the sample.

    Args:
        meta_keys (Sequence[str]): The meta keys to be saved in the
            ``metainfo`` of the packed ``data_samples``.
            Defaults to a tuple includes keys:

            - ``sample_idx``: The id of the image sample.
            - ``img_path``: The path to the image file.
            - ``ori_shape``: The original shape of the image as a tuple (H, W).
            - ``img_shape``: The shape of the image after the pipeline as a
              tuple (H, W).
            - ``scale_factor``: The scale factor between the resized image and
              the original image.
            - ``flip``: A boolean indicating if image flip transform was used.
            - ``flip_direction``: The flipping direction.
    """

    def __init__(self,
                 meta_keys=('sample_idx', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data."""
        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            packed_results['inputs'] = to_tensor(img)

        data_sample = ClsDataSample()
        if 'gt_label' in results:
            gt_label = results['gt_label']
            data_sample.set_gt_label(gt_label)

        img_meta = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str
