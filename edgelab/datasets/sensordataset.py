import os
import glob
import json

from typing import List, Optional, Sequence, Tuple, Union
from edgelab.registry import DATASETS
from mmcls.datasets import CustomDataset

import numpy as np


@DATASETS.register_module()
class SensorDataset(CustomDataset):
    CLASSES = []
    
    def __init__(self,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_root: str = '',
                 data_prefix: Union[str, dict] = '',
                 multi_label: bool = False,
                 **kwargs):
        
        if multi_label:
            raise NotImplementedError(
                'The `multi_label` option is not supported by now.')
        self.multi_label = multi_label
        self.data_root = data_root
        self.ann_file = ann_file
        self.data_prefix = data_prefix
        
        self.info_lables = json.load(
            open(os.path.join(self.data_root, self.data_prefix, self.ann_file)))
        

        for i in range(len(self.info_lables['files'])):
            if self.info_lables['files'][i]['label']['label'] not in self.CLASSES:
                self.CLASSES.append(
                    self.info_lables['files'][i]['label']['label'])
        
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=data_prefix,
            **kwargs)
        

    def get_classes(self, classes=None):

        if classes is not None:
            return classes

        class_names = []

        for i in range(len(self.info_lables['files'])):
            if self.info_lables['files'][i]['label']['label'] not in class_names:
                class_names.append(
                    self.info_lables['files'][i]['label']['label'])
       
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
        print(samples)
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
            img_path = os.path.join(self.img_prefix, filename)
            info = {'file_path': img_path, 'gt_label': int(gt_label)}
            data_list.append(info)
            
        return data_list

    def is_valid_file(self, filename: str) -> bool:
        """Check if a file is a valid sample."""
        return True
