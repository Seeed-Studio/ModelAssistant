# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import json
import os.path as osp
from typing import Optional, Sequence, Any

from mmdet.datasets import BaseDetDataset, CocoDataset

from sscma.registry import DATASETS, TASK_UTILS


class BatchShapePolicyDataset(BaseDetDataset):
    """Dataset with the batch shape policy that makes paddings with least
    pixels during batch inference process, which does not require the image
    scales of all batches to be the same throughout validation."""

    def __init__(self,
                 *args,
                 batch_shapes_cfg: Optional[dict] = None,
                 **kwargs):
        self.batch_shapes_cfg = batch_shapes_cfg
        super().__init__(*args, **kwargs)

    def full_init(self):
        """rewrite full_init() to be compatible with serialize_data in
        BatchShapePolicy."""
        if self._fully_initialized:
            return
        # load data information
        self.data_list = self.load_data_list()

        # batch_shapes_cfg
        if self.batch_shapes_cfg:
            batch_shapes_policy = TASK_UTILS.build(self.batch_shapes_cfg)
            self.data_list = batch_shapes_policy(self.data_list)
            del batch_shapes_policy

        # filter illegal data, such as data that has no annotations.
        self.data_list = self.filter_data()
        # Get subset data according to indices.
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)

        # serialize data_list
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()

        self._fully_initialized = True

    def prepare_data(self, idx: int) -> Any:
        """Pass the dataset to the pipeline during training to support mixed
        data augmentation, such as Mosaic and MixUp."""
        if self.test_mode is False:
            data_info = self.get_data_info(idx)
            data_info['dataset'] = self
            return self.pipeline(data_info)
        else:
            return super().prepare_data(idx)


@DATASETS.register_module()
class CustomYOLOv5CocoDataset(BatchShapePolicyDataset, CocoDataset):
    METAINFO = {
        'classes': (),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [
            (220, 20, 60),
            (119, 11, 32),
            (0, 0, 142),
            (0, 0, 230),
            (106, 0, 228),
            (0, 60, 100),
            (0, 80, 100),
            (0, 0, 70),
            (0, 0, 192),
            (250, 170, 30),
            (100, 170, 30),
            (220, 220, 0),
            (175, 116, 175),
            (250, 0, 30),
            (165, 42, 42),
            (255, 77, 255),
            (0, 226, 252),
            (182, 182, 255),
            (0, 82, 0),
            (120, 166, 157),
            (110, 76, 0),
            (174, 57, 255),
            (199, 100, 0),
            (72, 0, 118),
            (255, 179, 240),
            (0, 125, 92),
            (209, 0, 151),
            (188, 208, 182),
            (0, 220, 176),
            (255, 99, 164),
            (92, 0, 73),
            (133, 129, 255),
            (78, 180, 255),
            (0, 228, 0),
            (174, 255, 243),
            (45, 89, 255),
            (134, 134, 103),
            (145, 148, 174),
            (255, 208, 186),
            (197, 226, 255),
            (171, 134, 1),
            (109, 63, 54),
            (207, 138, 255),
            (151, 0, 95),
            (9, 80, 61),
            (84, 105, 51),
            (74, 65, 105),
            (166, 196, 102),
            (208, 195, 210),
            (255, 109, 65),
            (0, 143, 149),
            (179, 0, 194),
            (209, 99, 106),
            (5, 121, 0),
            (227, 255, 205),
            (147, 186, 208),
            (153, 69, 1),
            (3, 95, 161),
            (163, 255, 0),
            (119, 0, 170),
            (0, 182, 199),
            (0, 165, 120),
            (183, 130, 88),
            (95, 32, 0),
            (130, 114, 135),
            (110, 129, 133),
            (166, 74, 118),
            (219, 142, 185),
            (79, 210, 114),
            (178, 90, 62),
            (65, 70, 15),
            (127, 167, 115),
            (59, 105, 106),
            (142, 108, 45),
            (196, 172, 0),
            (95, 54, 80),
            (128, 76, 255),
            (201, 57, 1),
            (246, 0, 122),
            (191, 162, 208),
        ],
    }

    def __init__(
        self,
        *args,
        ann_file: str = '',
        metainfo: Optional[dict] = None,
        data_root: str = '',
        filter_supercat: bool = True,
        batch_shapes_cfg: Optional[dict] = None,
        classes: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        if metainfo is None and not self.METAINFO['classes']:
            if not osp.isabs(ann_file) and ann_file:
                self.ann_file = osp.join(data_root, ann_file)
            with open(self.ann_file, 'r') as f:
                data = json.load(f)
            if filter_supercat:
                categories = tuple(
                    cat['name'] for cat in data['categories'] if cat.get('supercategory', None) != 'none'
                )
            else:
                categories = tuple(cat['name'] for cat in data['categories'])
            self.METAINFO['classes'] = categories
        elif classes:
            self.METAINFO['classes'] = classes

        super().__init__(
            *args,
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            batch_shapes_cfg=batch_shapes_cfg,
            **kwargs,
        )