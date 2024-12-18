# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch
import os.path as osp
import numpy as np
from typing import List, Union, Sequence

from mmengine.fileio import get_local_path
from mmengine.dist import get_dist_info

from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


def coco_collate(data_batch: Sequence, use_ms_training: bool = False) -> dict:
    """Rewrite collate_fn to get faster training speed.

    Args:
       data_batch (Sequence): Batch of data.
       use_ms_training (bool): Whether to use multi-scale training.
    """
    batch_imgs = []
    batch_bboxes_labels = []
    batch_masks = []
    batch_keyponits = []
    batch_keypoints_visible = []
    for i in range(len(data_batch)):
        datasamples = data_batch[i]["data_samples"]
        inputs = data_batch[i]["inputs"]
        batch_imgs.append(inputs)

        gt_bboxes = datasamples.gt_instances.bboxes.tensor
        gt_labels = datasamples.gt_instances.labels
        if "masks" in datasamples.gt_instances:
            masks = datasamples.gt_instances.masks
            batch_masks.append(masks)
        if "gt_panoptic_seg" in datasamples:
            batch_masks.append(datasamples.gt_panoptic_seg.pan_seg)
        if "keypoints" in datasamples.gt_instances:
            keypoints = datasamples.gt_instances.keypoints
            keypoints_visible = datasamples.gt_instances.keypoints_visible
            batch_keyponits.append(keypoints)
            batch_keypoints_visible.append(keypoints_visible)

        batch_idx = gt_labels.new_full((len(gt_labels), 1), i)
        bboxes_labels = torch.cat((batch_idx, gt_labels[:, None], gt_bboxes), dim=1)
        batch_bboxes_labels.append(bboxes_labels)
    collated_results = {
        "data_samples": {"bboxes_labels": torch.cat(batch_bboxes_labels, 0)}
    }
    if len(batch_masks) > 0:
        collated_results["data_samples"]["masks"] = torch.cat(batch_masks, 0)

    if len(batch_keyponits) > 0:
        collated_results["data_samples"]["keypoints"] = torch.cat(batch_keyponits, 0)
        collated_results["data_samples"]["keypoints_visible"] = torch.cat(
            batch_keypoints_visible, 0
        )

    if use_ms_training:
        collated_results["inputs"] = batch_imgs
    else:
        collated_results["inputs"] = torch.stack(batch_imgs, 0)
    return collated_results


class BatchShapePolicy:
    """BatchShapePolicy is only used in the testing phase, which can reduce the
    number of pad pixels during batch inference.

    Args:
       batch_size (int): Single GPU batch size during batch inference.
           Defaults to 32.
       img_size (int): Expected output image size. Defaults to 640.
       size_divisor (int): The minimum size that is divisible
           by size_divisor. Defaults to 32.
       extra_pad_ratio (float):  Extra pad ratio. Defaults to 0.5.
    """

    def __init__(
        self,
        batch_size: int = 32,
        img_size: int = 640,
        size_divisor: int = 32,
        extra_pad_ratio: float = 0.5,
    ):
        self.img_size = img_size
        self.size_divisor = size_divisor
        self.extra_pad_ratio = extra_pad_ratio
        _, world_size = get_dist_info()
        # During multi-gpu testing, the batchsize should be multiplied by
        # worldsize, so that the number of batches can be calculated correctly.
        # The index of batches will affect the calculation of batch shape.
        self.batch_size = batch_size * world_size

    def __call__(self, data_list: List[dict]) -> List[dict]:
        image_shapes = []
        for data_info in data_list:
            image_shapes.append((data_info["width"], data_info["height"]))

        image_shapes = np.array(image_shapes, dtype=np.float64)

        n = len(image_shapes)  # number of images
        batch_index = np.floor(np.arange(n) / self.batch_size).astype(
            np.int64
        )  # batch index
        number_of_batches = batch_index[-1] + 1  # number of batches

        aspect_ratio = image_shapes[:, 1] / image_shapes[:, 0]  # aspect ratio
        irect = aspect_ratio.argsort()

        data_list = [data_list[i] for i in irect]

        aspect_ratio = aspect_ratio[irect]
        # Set training image shapes
        shapes = [[1, 1]] * number_of_batches
        for i in range(number_of_batches):
            aspect_ratio_index = aspect_ratio[batch_index == i]
            min_index, max_index = aspect_ratio_index.min(), aspect_ratio_index.max()
            if max_index < 1:
                shapes[i] = [max_index, 1]
            elif min_index > 1:
                shapes[i] = [1, 1 / min_index]

        batch_shapes = (
            np.ceil(
                np.array(shapes) * self.img_size / self.size_divisor
                + self.extra_pad_ratio
            ).astype(np.int64)
            * self.size_divisor
        )

        for i, data_info in enumerate(data_list):
            data_info["batch_shape"] = batch_shapes[batch_index[i]]

        return data_list


class CocoDataset(BaseDetDataset):
    """Dataset for COCO."""

    METAINFO = {
        "classes": (
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ),
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
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
            self.ann_file, backend_args=self.backend_args
        ) as local_path:
            self.coco = self.COCOAPI(local_path)
            
        self._metainfo["classes"] = set([cat["name"] for cat in self.coco.cats.values()]) if len(self.coco.cats.values()) > 0 else self.METAINFO["classes"]
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.metainfo["classes"])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

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

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info["raw_img_info"]
        ann_info = raw_data_info["raw_ann_info"]

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix["img"], img_info["file_name"])
        if self.data_prefix.get("seg", None):
            seg_map_path = osp.join(
                self.data_prefix["seg"],
                img_info["file_name"].rsplit(".", 1)[0] + self.seg_map_suffix,
            )
        else:
            seg_map_path = None
        data_info["img_path"] = img_path
        data_info["img_id"] = img_info["img_id"]
        data_info["seg_map_path"] = seg_map_path
        data_info["height"] = img_info["height"]
        data_info["width"] = img_info["width"]

        if self.return_classes:
            data_info["text"] = self.metainfo["classes"]
            data_info["caption_prompt"] = self.caption_prompt
            data_info["custom_entities"] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get("iscrowd", False):
                instance["ignore_flag"] = 1
            else:
                instance["ignore_flag"] = 0
            instance["bbox"] = bbox
            instance["bbox_label"] = self.cat2label[ann["category_id"]]

            if ann.get("segmentation", None):
                instance["mask"] = ann["segmentation"]

            instances.append(instance)
        data_info["instances"] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get("filter_empty_gt", False)
        min_size = self.filter_cfg.get("min_size", 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info["img_id"] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info["img_id"]
            width = data_info["width"]
            height = data_info["height"]
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
