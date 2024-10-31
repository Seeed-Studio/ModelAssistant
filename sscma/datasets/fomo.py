# Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) OpenMMLab.
import json
import os.path as osp
from typing import Optional, Sequence, List

from mmdet.datasets import CocoDataset
from mmengine.fileio import get_local_path
from .utils.download import check_file


class CustomFomoCocoDataset(CocoDataset):
    METAINFO = {
        "classes": (),
        # palette is a list of color tuples, which is used for visualization.
        "palette": [
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
        data_prefix: dict = dict(img_path=""),
        ann_file: str = "",
        metainfo: Optional[dict] = None,
        data_root: str = "",
        filter_supercat: bool = True,
        classes: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        if data_root:
            if not (osp.isabs(ann_file) and (osp.isabs(data_prefix.get("img", "")))):
                data_root = (
                    check_file(data_root, data_name="coco") if data_root else data_root
                )
        if metainfo is None and not self.METAINFO["classes"]:
            if not osp.isabs(ann_file) and ann_file:
                self.ann_file = osp.join(data_root, ann_file)
            with open(self.ann_file, "r") as f:
                data = json.load(f)
            if filter_supercat:
                categories = tuple(
                    cat["name"]
                    for cat in data["categories"]
                    if cat["supercategory"] != "none"
                )
            else:
                categories = tuple(cat["name"] for cat in data["categories"])
            self.METAINFO["classes"] = categories
        elif classes:
            self.METAINFO["classes"] = classes

        super().__init__(
            *args,
            data_prefix=data_prefix,
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            **kwargs,
        )

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501

        with get_local_path(
            self.ann_file, backend_args=self.backend_args
        ) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=(
                self.metainfo["classes"]
                if len(self.metainfo["classes"])
                else [
                    cat["name"]
                    for cat in self.coco.dataset["categories"]
                    if (cat["supercategory"] != "none")
                ]
            ),
            sup_names=[
                cat["supercategory"]
                for cat in self.coco.dataset["categories"]
                if (cat["supercategory"] != "none")
            ],
        )
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = self.coco.cat_img_map

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info["img_id"] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info(
                {"raw_ann_info": raw_ann_info, "raw_img_info": raw_img_info}
            )
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
