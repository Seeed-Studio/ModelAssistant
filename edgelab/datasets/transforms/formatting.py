from collections import defaultdict
from collections.abc import Sequence
from functools import partial

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.utils import is_str
from PIL import Image

from edgelab.registry import TRANSFORMS
from mmcls.structures import ClsDataSample

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
            '`Sequence`, `int` and `float`')


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
