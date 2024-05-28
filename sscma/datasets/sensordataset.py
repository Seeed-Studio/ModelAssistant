# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import json
import os
import os.path as osp
from typing import List, Optional, Union

import cbor
import numpy as np

from sscma.registry import DATASETS

from .clsdataset import CustomClsDataset


@DATASETS.register_module()
class SensorDataset(CustomClsDataset):
    CLASSES = []

    def __init__(
        self,
        ann_file: str = '',
        metainfo: Optional[dict] = None,
        data_root: str = '',
        data_prefix: Union[str, dict] = '',
        window_size: int = 80,
        stride: int = 30,
        retention: float = 0.8,
        # source: str = 'EI',
        flatten: bool = True,
        multi_label: bool = False,
        **kwargs,
    ):
        if multi_label:
            raise NotImplementedError('The `multi_label` option is not supported by now.')
        self.multi_label = multi_label
        self.data_root = data_root
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        self.window_size = window_size
        self.stride = stride
        self.retention = retention
        self.flatten = flatten

        self.data_dir = osp.join(self.data_root, self.data_prefix)

        self.info_lables = json.load(open(os.path.join(self.data_root, self.data_prefix, self.ann_file)))

        for i in range(len(self.info_lables['files'])):
            if self.info_lables['files'][i]['label']['label'] not in self.CLASSES:
                self.CLASSES.append(self.info_lables['files'][i]['label']['label'])

        super().__init__(ann_file=ann_file, metainfo=metainfo, data_root=data_root, data_prefix=data_prefix, **kwargs)

        self.metainfo = {'classes': self.get_classes()}

    def get_classes(self, classes=None):
        if classes is not None:
            return classes

        class_names = []

        for i in range(len(self.info_lables['files'])):
            if self.info_lables['files'][i]['label']['label'] not in class_names:
                class_names.append(self.info_lables['files'][i]['label']['label'])

        return class_names

    def _find_samples(self):
        samples = []
        for i in range(len(self.info_lables['files'])):
            filename = self.info_lables['files'][i]['path']
            gt_label = 0

            for j in range(len(self.CLASSES)):
                if self.CLASSES[j] == self.info_lables['files'][i]['label']['label']:
                    gt_label = j
                    break
            samples.append((filename, gt_label))
        return samples

    def load_data_list(self):
        samples = []
        for i in range(len(self.info_lables['files'])):
            filename = self.info_lables['files'][i]['path']
            gt_label = 0

            for j in range(len(self.CLASSES)):
                if self.CLASSES[j] == self.info_lables['files'][i]['label']['label']:
                    gt_label = j
                    break
            samples.append((filename, gt_label))

        data_list = []
        for filename, gt_label in samples:
            ann_path = os.path.join(self.data_dir, filename)
            sensors, data_set = self.read_split_data(ann_path)
            data_list.extend(
                [{'data': np.asanyarray([data]), 'gt_label': int(gt_label), 'sensors': sensors} for data in data_set]
            )

        return data_list

    def read_split_data(self, file_path: str) -> List:
        if file_path.lower().endswith('.cbor'):
            with open(file_path, 'rb') as f:
                info_lables = cbor.loads(f.read())
        elif file_path.lower().endswith('.json'):
            with open(file_path, 'r') as f:
                info_lables = json.load(f)

        values = np.asanyarray(info_lables['payload']['values'])
        sensors = info_lables['payload']['sensors']

        data_set = []
        values_len = len(values)
        if values_len <= self.window_size:
            data_set.append(self.pad_data(values, self.window_size).transpose(0, 1).reshape(-1))
        else:
            indexes = range(0, values_len, self.stride)
            for i in indexes:
                if (values_len - i + 1) < self.window_size or i == indexes[-1]:
                    if self.retention * self.window_size < (values_len - i + 1):
                        data = self.pad_data(values[i:], self.window_size)
                    else:
                        continue
                else:
                    end = i + self.window_size
                    if end >= values_len:
                        if self.retention * self.window_size < (values_len - i + 1):
                            data = self.pad_data(values[i:], self.window_size)
                        else:
                            continue
                    else:
                        data = values[i:end]
                if self.flatten:
                    data = data.transpose(0, 1).reshape(-1)
                data_set.append(data)

        return sensors, data_set

    def pad_data(self, data: np.asanyarray, total_len: int, mode='constant', pad_val=0) -> np.array:
        pad_len = total_len - len(data)
        front = pad_len // 2
        arfter = pad_len - front
        data = np.pad(data, ((front, arfter), (0, 0)), mode=mode, constant_values=pad_val)
        return data

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return True
