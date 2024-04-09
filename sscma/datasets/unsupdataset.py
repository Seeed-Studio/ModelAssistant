# Copyright (c) Seeed Tech Ltd. All rights reserved.
import os
import os.path as osp
from typing import List, Optional, Sequence, Union

from mmengine.dataset import BaseDataset

from sscma.registry import DATASETS


@DATASETS.register_module()
class UnsupDataset(BaseDataset):
    METAINFO: dict = dict(
        classes=(),
        # palette is a list of color tuples, which is used for visualization.
        palette=[
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
    )

    def __init__(
        self,
        ann_file: str = '',
        metainfo: Optional[dict] = None,
        data_root: str = '',
        data_prefix: dict = dict(img=''),
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        serialize_data: bool = True,
        pipeline=None,
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
    ):
        super().__init__(
            ann_file,
            metainfo,
            data_root,
            data_prefix,
            filter_cfg,
            indices,
            serialize_data,
            pipeline,
            test_mode,
            lazy_init,
            max_refetch,
        )

        if data_root and data_prefix and not osp.isabs(data_prefix['img']):
            self.data_prefix = osp.join(data_root, data_prefix['img'])
        elif osp.isabs(data_prefix['img']):
            self.data_prefix = data_prefix['img']
        else:
            raise ValueError
        print(self.data_prefix)

    def load_data_list(self) -> List[dict]:
        print(self.data_prefix)
        filels = [osp.join(self.data_prefix['img'], i) for i in os.listdir(self.data_prefix['img'])]
        data_list = []
        for idx, file in enumerate(filels):
            # img = cv2.imread(file)
            data_list.append({'img_path': file, 'img_id': idx})
        return data_list

    def __getitem__(self, idx: int) -> dict:
        return super().__getitem__(idx)

    @property
    def metainfo(self) -> dict:
        return self.METAINFO

    @metainfo.setter
    def metainfo(self, metainfo) -> None:
        self.METAINFO = metainfo
