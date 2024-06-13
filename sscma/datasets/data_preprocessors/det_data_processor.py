from typing import Union

import torch

from mmengine.structures import BaseDataElement
from mmdet.models.data_preprocessors import DetDataPreprocessor as MMDetDataPreprocessor

from sscma.registry import MODELS

CastData = Union[tuple, dict, BaseDataElement, torch.Tensor, list, bytes, str, None]


@MODELS.register_module()
class DetDataPreprocessor(MMDetDataPreprocessor):
    """
    Optimized data casting function for the detector model, only works for a specified input format.
    Actually the most time wasting part on data casting is the data transfer between CPU and GPU.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # overide the 'cast_data' function if it exists in base class
        if hasattr(super(), 'cast_data'):
            if self.device.type == 'cuda':
                self.cast_data = self._cast_data_cuda
            elif self.device.type == 'mps':
                self.cast_data = self._cast_data_mps
            else:
                self.cast_data = self._cast_data_default

    def _cast_data_cuda(self, data: CastData):
        data_dict = {
            'inputs': [tsr.to(self.device, non_blocking=True) for tsr in data['inputs']],
            'data_samples': [smp.to(self.device, non_blocking=True) for smp in data['data_samples']],
        }

        if not self._non_blocking:
            torch.cuda.synchronize(self.device)

        return data_dict

    def _cast_data_mps(self, data: CastData):
        data_dict = {
            'inputs': [tsr.to(self.device, non_blocking=True) for tsr in data['inputs']],
            'data_samples': [smp.to(self.device, non_blocking=True) for smp in data['data_samples']],
        }

        if not self._non_blocking:
            torch.mps.synchronize()

        return data_dict

    def _cast_data_default(self, data: CastData):
        data_dict = {
            'inputs': [tsr.to(self.device, non_blocking=self._non_blocking) for tsr in data['inputs']],
            'data_samples': [smp.to(self.device, non_blocking=self._non_blocking) for smp in data['data_samples']],
        }

        return data_dict
