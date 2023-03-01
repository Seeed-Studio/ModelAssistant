import os
import glob
import json

from typing import List, Optional, Sequence, Tuple, Union
from mmcls.datasets.builder import DATASETS
from mmcls.datasets.custom import CustomDataset


import numpy as np


@DATASETS.register_module()
class AxesDataset(CustomDataset):
    CLASSES = []

    def __init__(self,
                 data_root: str,
                 label_file: Optional[str] = None,
                 test_mode=False,
                 classes=None,
                 mode: str = 'train',
                 ):

        self.info_lables = None
        self.mode = mode
        self.data_root = data_root
        self.label_file = label_file
        self.info_lables = json.load(
            open(os.path.join(self.data_root, self.label_file)))

        super().__init__(data_prefix=data_root, test_mode=test_mode,
                         classes=classes, ann_file=label_file)

    def get_classes(self, classes=None):

        if classes is not None:
            return classes

        class_names = []

        for i in range(len(self.info_lables['files'])):
            if self.info_lables['files'][i]['label']['label'] not in class_names:
                class_names.append(
                    self.info_lables['files'][i]['label']['label'])
       
        return class_names

    def prepare_data(self, idx):

        data = np.array([], dtype=np.float32)

        ds = json.load(
            open(os.path.join(os.path.dirname(os.path.join(self.data_root, self.label_file)), self.info_lables['files'][idx]['path'])))

        for d in ds['payload']['values']:
            data = np.append(data, np.array(d, dtype=np.float32))
        
        if self.mode == 'train':
            return {'img': data, 'gt_label': self.data_infos[idx]['gt_label']}

        return {'img': data}

    def load_annotations(self):
        """Load axes paths and gt_labels."""
        data_infos = []
        for i in range(len(self.info_lables['files'])):
            filename = self.info_lables['files'][i]['path']
            gt_label = 0

            for j in range(len(self.CLASSES)):
                if self.CLASSES[j] == self.info_lables['files'][i]['label']['label']:
                    gt_label = j
                    break;

            info = {'img_prefix': self.data_root}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            
            data_infos.append(info)

        return data_infos
