# copyright Copyright (c) Seeed Technology Co.,Ltd.
from typing import Union

from mmengine.structures import BaseDataElement, InstanceData, PixelData

from .multilevel_pixel_data import MultilevelPixelData


class PoseDataSample(BaseDataElement):
    @property
    def gt_instances(self) -> InstanceData:
        return self._gt_instances

    @gt_instances.setter
    def gt_instances(self, value: InstanceData):
        self.set_field(value, '_gt_instances', dtype=InstanceData)

    @gt_instances.deleter
    def gt_instances(self):
        del self._gt_instances

    @property
    def gt_instance_labels(self) -> InstanceData:
        return self._gt_instance_labels

    @gt_instance_labels.setter
    def gt_instance_labels(self, value: InstanceData):
        self.set_field(value, '_gt_instance_labels', dtype=InstanceData)

    @gt_instance_labels.deleter
    def gt_instance_labels(self):
        del self._gt_instance_labels

    @property
    def pred_instances(self) -> InstanceData:
        return self._pred_instances

    @pred_instances.setter
    def pred_instances(self, value: InstanceData):
        self.set_field(value, '_pred_instances', dtype=InstanceData)

    @pred_instances.deleter
    def pred_instances(self):
        del self._pred_instances

    @property
    def gt_fields(self) -> Union[PixelData, MultilevelPixelData]:
        return self._gt_fields

    @gt_fields.setter
    def gt_fields(self, value: Union[PixelData, MultilevelPixelData]):
        self.set_field(value, '_gt_fields', dtype=type(value))

    @gt_fields.deleter
    def gt_fields(self):
        del self._gt_fields

    @property
    def pred_fields(self) -> PixelData:
        return self._pred_heatmaps

    @pred_fields.setter
    def pred_fields(self, value: PixelData):
        self.set_field(value, '_pred_heatmaps', dtype=PixelData)

    @pred_fields.deleter
    def pred_fields(self):
        del self._pred_heatmaps
