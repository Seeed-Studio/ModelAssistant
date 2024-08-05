# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing.reduction import ForkingPickler
from typing import Union, List, Optional

import numpy as np
import torch
from mmengine.structures import BaseDataElement, InstanceData, PixelData

from .utils import LABEL_TYPE, SCORE_TYPE, format_label, format_score


class MultiTaskDataSample(BaseDataElement):
    @property
    def tasks(self):
        return self._data_fields


class DataSample(BaseDataElement):
    """A general data structure interface.

    It's used as the interface between different components.

    The following fields are convention names in MMPretrain, and we will set or
    get these fields in data transforms, models, and metrics if needed. You can
    also set any new fields for your need.

    Meta fields:
        img_shape (Tuple): The shape of the corresponding input image.
        ori_shape (Tuple): The original shape of the corresponding image.
        sample_idx (int): The index of the sample in the dataset.
        num_classes (int): The number of all categories.

    Data fields:
        gt_label (tensor): The ground truth label.
        gt_score (tensor): The ground truth score.
        pred_label (tensor): The predicted label.
        pred_score (tensor): The predicted score.
        mask (tensor): The mask used in masked image modeling.

    Examples:
        >>> import torch
        >>> from mmpretrain.structures import DataSample
        >>>
        >>> img_meta = dict(img_shape=(960, 720), num_classes=5)
        >>> data_sample = DataSample(metainfo=img_meta)
        >>> data_sample.set_gt_label(3)
        >>> print(data_sample)
        <DataSample(
        META INFORMATION
            num_classes: 5
            img_shape: (960, 720)
        DATA FIELDS
            gt_label: tensor([3])
        ) at 0x7ff64c1c1d30>
        >>>
        >>> # For multi-label data
        >>> data_sample = DataSample().set_gt_label([0, 1, 4])
        >>> print(data_sample)
        <DataSample(
        DATA FIELDS
            gt_label: tensor([0, 1, 4])
        ) at 0x7ff5b490e100>
        >>>
        >>> # Set one-hot format score
        >>> data_sample = DataSample().set_pred_score([0.1, 0.1, 0.6, 0.1])
        >>> print(data_sample)
        <DataSample(
        META INFORMATION
            num_classes: 4
        DATA FIELDS
            pred_score: tensor([0.1000, 0.1000, 0.6000, 0.1000])
        ) at 0x7ff5b48ef6a0>
        >>>
        >>> # Set custom field
        >>> data_sample = DataSample()
        >>> data_sample.my_field = [1, 2, 3]
        >>> print(data_sample)
        <DataSample(
        DATA FIELDS
            my_field: [1, 2, 3]
        ) at 0x7f8e9603d3a0>
        >>> print(data_sample.my_field)
        [1, 2, 3]
    """

    def set_gt_label(self, value: LABEL_TYPE) -> "DataSample":
        """Set ``gt_label``."""
        self.set_field(format_label(value), "gt_label", dtype=torch.Tensor)
        return self

    def set_gt_score(self, value: SCORE_TYPE) -> "DataSample":
        """Set ``gt_score``."""
        score = format_score(value)
        self.set_field(score, "gt_score", dtype=torch.Tensor)
        if hasattr(self, "num_classes"):
            assert len(score) == self.num_classes, (
                f"The length of score {len(score)} should be "
                f"equal to the num_classes {self.num_classes}."
            )
        else:
            self.set_field(name="num_classes", value=len(score), field_type="metainfo")
        return self

    def set_pred_label(self, value: LABEL_TYPE) -> "DataSample":
        """Set ``pred_label``."""
        self.set_field(format_label(value), "pred_label", dtype=torch.Tensor)
        return self

    def set_pred_score(self, value: SCORE_TYPE):
        """Set ``pred_label``."""
        score = format_score(value)
        self.set_field(score, "pred_score", dtype=torch.Tensor)
        if hasattr(self, "num_classes"):
            assert len(score) == self.num_classes, (
                f"The length of score {len(score)} should be "
                f"equal to the num_classes {self.num_classes}."
            )
        else:
            self.set_field(name="num_classes", value=len(score), field_type="metainfo")
        return self

    def set_mask(self, value: Union[torch.Tensor, np.ndarray]):
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        elif not isinstance(value, torch.Tensor):
            raise TypeError(f"Invalid mask type {type(value)}")
        self.set_field(value, "mask", dtype=torch.Tensor)
        return self

    def __repr__(self) -> str:
        """Represent the object."""

        def dump_items(items, prefix=""):
            return "\n".join(f"{prefix}{k}: {v}" for k, v in items)

        repr_ = ""
        if len(self._metainfo_fields) > 0:
            repr_ += "\n\nMETA INFORMATION\n"
            repr_ += dump_items(self.metainfo_items(), prefix=" " * 4)
        if len(self._data_fields) > 0:
            repr_ += "\n\nDATA FIELDS\n"
            repr_ += dump_items(self.items(), prefix=" " * 4)

        repr_ = f"<{self.__class__.__name__}({repr_}\n\n) at {hex(id(self))}>"
        return repr_


def _reduce_datasample(data_sample):
    """reduce DataSample."""
    attr_dict = data_sample.__dict__
    convert_keys = []
    for k, v in attr_dict.items():
        if isinstance(v, torch.Tensor):
            attr_dict[k] = v.numpy()
            convert_keys.append(k)
    return _rebuild_datasample, (attr_dict, convert_keys)


def _rebuild_datasample(attr_dict, convert_keys):
    """rebuild DataSample."""
    data_sample = DataSample()
    for k in convert_keys:
        attr_dict[k] = torch.from_numpy(attr_dict[k])
    data_sample.__dict__ = attr_dict
    return data_sample


# Due to the multi-processing strategy of PyTorch, DataSample may consume many
# file descriptors because it contains multiple tensors. Here we overwrite the
# reduce function of DataSample in ForkingPickler and convert these tensors to
# np.ndarray during pickling. It may slightly influence the performance of
# dataloader.
ForkingPickler.register(DataSample, _reduce_datasample)


class DetDataSample(BaseDataElement):
    """A data structure interface of MMDetection. They are used as interfaces
    between different components.

    The attributes in ``DetDataSample`` are divided into several parts:

        - ``proposals``(InstanceData): Region proposals used in two-stage
            detectors.
        - ``gt_instances``(InstanceData): Ground truth of instance annotations.
        - ``pred_instances``(InstanceData): Instances of model predictions.
        - ``ignored_instances``(InstanceData): Instances to be ignored during
            training/testing.
        - ``gt_panoptic_seg``(PixelData): Ground truth of panoptic
            segmentation.
        - ``pred_panoptic_seg``(PixelData): Prediction of panoptic
           segmentation.
        - ``gt_sem_seg``(PixelData): Ground truth of semantic segmentation.
        - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.

    Examples:
         >>> import torch
         >>> import numpy as np
         >>> from mmengine.structures import InstanceData
         >>> from mmdet.structures import DetDataSample

         >>> data_sample = DetDataSample()
         >>> img_meta = dict(img_shape=(800, 1196),
         ...                 pad_shape=(800, 1216))
         >>> gt_instances = InstanceData(metainfo=img_meta)
         >>> gt_instances.bboxes = torch.rand((5, 4))
         >>> gt_instances.labels = torch.rand((5,))
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'img_shape' in data_sample.gt_instances.metainfo_keys()
         >>> len(data_sample.gt_instances)
         5
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            gt_instances: <InstanceData(

                    META INFORMATION
                    pad_shape: (800, 1216)
                    img_shape: (800, 1196)

                    DATA FIELDS
                    labels: tensor([0.8533, 0.1550, 0.5433, 0.7294, 0.5098])
                    bboxes:
                    tensor([[9.7725e-01, 5.8417e-01, 1.7269e-01, 6.5694e-01],
                            [1.7894e-01, 5.1780e-01, 7.0590e-01, 4.8589e-01],
                            [7.0392e-01, 6.6770e-01, 1.7520e-01, 1.4267e-01],
                            [2.2411e-01, 5.1962e-01, 9.6953e-01, 6.6994e-01],
                            [4.1338e-01, 2.1165e-01, 2.7239e-04, 6.8477e-01]])
                ) at 0x7f21fb1b9190>
        ) at 0x7f21fb1b9880>
         >>> pred_instances = InstanceData(metainfo=img_meta)
         >>> pred_instances.bboxes = torch.rand((5, 4))
         >>> pred_instances.scores = torch.rand((5,))
         >>> data_sample = DetDataSample(pred_instances=pred_instances)
         >>> assert 'pred_instances' in data_sample

         >>> data_sample = DetDataSample()
         >>> gt_instances_data = dict(
         ...                        bboxes=torch.rand(2, 4),
         ...                        labels=torch.rand(2),
         ...                        masks=np.random.rand(2, 2, 2))
         >>> gt_instances = InstanceData(**gt_instances_data)
         >>> data_sample.gt_instances = gt_instances
         >>> assert 'gt_instances' in data_sample
         >>> assert 'masks' in data_sample.gt_instances

         >>> data_sample = DetDataSample()
         >>> gt_panoptic_seg_data = dict(panoptic_seg=torch.rand(2, 4))
         >>> gt_panoptic_seg = PixelData(**gt_panoptic_seg_data)
         >>> data_sample.gt_panoptic_seg = gt_panoptic_seg
         >>> print(data_sample)
        <DetDataSample(

            META INFORMATION

            DATA FIELDS
            _gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
            gt_panoptic_seg: <BaseDataElement(

                    META INFORMATION

                    DATA FIELDS
                    panoptic_seg: tensor([[0.7586, 0.1262, 0.2892, 0.9341],
                                [0.3200, 0.7448, 0.1052, 0.5371]])
                ) at 0x7f66c2bb7730>
        ) at 0x7f66c2bb7280>
        >>> data_sample = DetDataSample()
        >>> gt_segm_seg_data = dict(segm_seg=torch.rand(2, 2, 2))
        >>> gt_segm_seg = PixelData(**gt_segm_seg_data)
        >>> data_sample.gt_segm_seg = gt_segm_seg
        >>> assert 'gt_segm_seg' in data_sample
        >>> assert 'segm_seg' in data_sample.gt_segm_seg
    """

    @property
    def proposals(self) -> InstanceData:
        return self._proposals

    @proposals.setter
    def proposals(self, value: InstanceData):
        self.set_field(value, "_proposals", dtype=InstanceData)

    @proposals.deleter
    def proposals(self):
        del self._proposals

    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, "_gt_instances", dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, "_pred_instances", dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def ignored_instances(self) -> InstanceData:
        return self._ignored_instances

    @ignored_instances.setter
    def ignored_instances(self, value: InstanceData):
        self.set_field(value, "_ignored_instances", dtype=InstanceData)

    @ignored_instances.deleter
    def ignored_instances(self):
        del self._ignored_instances

    @property
    def gt_panoptic_seg(self) -> PixelData:
        return self._gt_panoptic_seg

    @gt_panoptic_seg.setter
    def gt_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_gt_panoptic_seg", dtype=PixelData)

    @gt_panoptic_seg.deleter
    def gt_panoptic_seg(self):
        del self._gt_panoptic_seg

    @property
    def pred_panoptic_seg(self) -> PixelData:
        return self._pred_panoptic_seg

    @pred_panoptic_seg.setter
    def pred_panoptic_seg(self, value: PixelData):
        self.set_field(value, "_pred_panoptic_seg", dtype=PixelData)

    @pred_panoptic_seg.deleter
    def pred_panoptic_seg(self):
        del self._pred_panoptic_seg

    @property
    def gt_sem_seg(self) -> PixelData:
        return self._gt_sem_seg

    @gt_sem_seg.setter
    def gt_sem_seg(self, value: PixelData):
        self.set_field(value, "_gt_sem_seg", dtype=PixelData)

    @gt_sem_seg.deleter
    def gt_sem_seg(self):
        del self._gt_sem_seg

    @property
    def pred_sem_seg(self) -> PixelData:
        return self._pred_sem_seg

    @pred_sem_seg.setter
    def pred_sem_seg(self, value: PixelData):
        self.set_field(value, "_pred_sem_seg", dtype=PixelData)

    @pred_sem_seg.deleter
    def pred_sem_seg(self):
        del self._pred_sem_seg


SampleList = List[DetDataSample]
OptSampleList = Optional[SampleList]
