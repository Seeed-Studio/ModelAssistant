# Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) OpenMMLab.
import math
import os
import urllib
from typing import List, Union

import numpy as np
import torch

from mmengine.utils import scandir
from sscma.structures import SampleList
from sscma.structures.bbox import BaseBoxes


IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


def samplelist_boxtype2tensor(batch_data_samples: SampleList) -> SampleList:
    for data_samples in batch_data_samples:
        if "gt_instances" in data_samples:
            bboxes = data_samples.gt_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.gt_instances.bboxes = bboxes.tensor
        if "pred_instances" in data_samples:
            bboxes = data_samples.pred_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.pred_instances.bboxes = bboxes.tensor
        if "ignored_instances" in data_samples:
            bboxes = data_samples.ignored_instances.get("bboxes", None)
            if isinstance(bboxes, BaseBoxes):
                data_samples.ignored_instances.bboxes = bboxes.tensor


def make_divisible(x: float, widen_factor: float = 1.0, divisor: int = 8) -> int:
    """Make sure that x*widen_factor is divisible by divisor."""
    return math.ceil(x * widen_factor / divisor) * divisor


def make_round(x: float, deepen_factor: float = 1.0) -> int:
    """Make sure that x*deepen_factor becomes an integer not less than 1."""
    return max(round(x * deepen_factor), 1) if x > 1 else x


def auto_arrange_images(image_list: list, image_column: int = 2) -> np.ndarray:
    """Auto arrange image to image_column x N row.

    Args:
        image_list (list): cv2 image list.
        image_column (int): Arrange to N column. Default: 2.
    Return:
        (np.ndarray): image_column x N row merge image
    """
    img_count = len(image_list)
    if img_count <= image_column:
        # no need to arrange
        image_show = np.concatenate(image_list, axis=1)
    else:
        # arrange image according to image_column
        image_row = round(img_count / image_column)
        fill_img_list = [np.ones(image_list[0].shape, dtype=np.uint8) * 255] * (
            image_row * image_column - img_count
        )
        image_list.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(image_row):
            start_col = image_column * i
            end_col = image_column * (i + 1)
            merge_col = np.hstack(image_list[start_col:end_col])
            merge_imgs_col.append(merge_col)

        # merge to one image
        image_show = np.vstack(merge_imgs_col)

    return image_show


def get_file_list(source_root: str) -> Union[List, dict]:
    """Get file list.

    Args:
        source_root (str): image or video source path

    Return:
        source_file_path_list (list): A list for all source file.
        source_type (dict): Source type: file or url or dir.
    """
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(("http:/", "https:/"))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        # when input source is dir
        for file in scandir(source_root, IMG_EXTENSIONS, recursive=True):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        # when input source is url
        filename = os.path.basename(urllib.parse.unquote(source_root).split("?")[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f"Downloading source file to {file_save_path}")
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        # when input source is single image
        source_file_path_list = [source_root]
    else:
        print("Cannot find image file.")

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)

    return source_file_path_list, source_type
