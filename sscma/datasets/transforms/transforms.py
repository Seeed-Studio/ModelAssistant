# Copyright (c) OpenMMLab. All rights reserved.
import copy
import collections
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Tuple, Union, Iterable, Dict, List

import cv2
import torch
import numpy as np
from numpy import random
import torchvision as tv
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.functional import InterpolationMode

import mmengine
from sscma.utils import *
from mmengine.dataset import BaseDataset
from sscma.structures.bbox import autocast_box_type
from sscma.utils.simplecv import simplecv_rescale_size, simplecv_imresize
from .utils import BaseTransform, cache_randomness

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

try:
    from PIL import Image
except ImportError:
    Image = None

Number = Union[int, float]

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

if Image is not None:
    if hasattr(Image, "Resampling"):
        pillow_interp_codes = {
            "nearest": Image.Resampling.NEAREST,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "box": Image.Resampling.BOX,
            "lanczos": Image.Resampling.LANCZOS,
            "hamming": Image.Resampling.HAMMING,
        }
    else:
        pillow_interp_codes = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "box": Image.BOX,
            "lanczos": Image.LANCZOS,
            "hamming": Image.HAMMING,
        }

Number = Union[int, float]


class ImageType:
    _torch = False
    _numpy = False

    @classmethod
    def get_torch(cls):
        return cls._torch

    @classmethod
    def set_torch(cls, value):
        cls._torch = value
        cls._numpy = False

    @classmethod
    def get_numpy(cls):
        return cls._numpy

    @classmethod
    def set_numpy(cls, value):
        cls._numpy = value
        cls._torch = False

    @property
    def numpy(self):
        return self.get_numpy()

    @property
    def torch(self):
        return self.get_torch()

    classmethod(property(get_numpy, set_numpy))
    classmethod(property(get_torch, set_torch))


class toTensor(BaseTransform):
    """
    image to pytorch tensor.
    """

    def transform(self, results: dict) -> dict:

        image = results["img"]
        img = F.to_dtype(F.to_image(image.copy()), torch.uint8, scale=True)

        # if img.size()[0] == 1:
        #     img = img.repeat(3, 1, 1)

        results["img"] = img
        results["torch"] = True
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class Resize(BaseTransform):
    """Resize images & bbox & seg & keypoints.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, seg map and keypoints are then resized with the
    same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(
        self,
        scale: Optional[Union[int, Tuple[int, int]]] = None,
        scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
        keep_ratio: bool = True,
        clip_object_border: bool = True,
        backend: str = "cv2",
        interpolation="bilinear",
    ) -> None:
        assert scale is not None or scale_factor is not None, (
            "`scale` and" "`scale_factor` can not both be `None`"
        )
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f"expect scale_factor is float or Tuple(float), but"
                f"get {type(scale_factor)}"
            )

    def imresize(
        self,
        img: np.ndarray,
        size: Tuple[int, int],
        return_scale: bool = False,
        interpolation: str = "bilinear",
        out: Optional[np.ndarray] = None,
        backend: Optional[str] = None,
    ) -> Union[Tuple[np.ndarray, float, float], np.ndarray]:
        """Resize image to a given size.

        Args:
            img (ndarray): The input image.
            size (tuple[int]): Target size (w, h).
            return_scale (bool): Whether to return `w_scale` and `h_scale`.
            interpolation (str): Interpolation method, accepted values are
                "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
                backend, "nearest", "bilinear" for 'pillow' backend.
            out (ndarray): The output destination.
            backend (str | None): The image resize backend type. Options are `cv2`,
                `pillow`, `None`. If backend is None, the global imread_backend
                specified by ``mmcv.use_backend()`` will be used. Default: None.

        Returns:
            tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
        """

        h, w = img.shape[:2] if isinstance(img, np.ndarray) else img.shape[1:]
        new_size, scale_factor = simplecv_rescale_size((w, h), size, return_scale=True)

        if isinstance(img, (torch.Tensor, tv.tv_tensors._image.Image)):
            backend = "torch"
        elif isinstance(img, np.ndarray):
            backend = "cv2"
        self.backend = backend
        if backend not in ["cv2", "pillow", "torch"]:
            raise ValueError(
                f"backend: {backend} is not supported for resize."
                f"Supported backends are 'cv2', 'pillow'"
            )
        if backend == "pillow":
            assert img.dtype == np.uint8, "Pillow backend only support uint8 type"
            pil_image = Image.fromarray(img)
            pil_image = pil_image.resize(new_size, pillow_interp_codes[interpolation])
            resized_img = np.array(pil_image)
        elif backend == "torch":
            resized_img = F.resize(
                img, new_size, interpolation=InterpolationMode.BILINEAR
            )
        else:
            resized_img = cv2.resize(
                img,
                new_size,
                dst=out,
                interpolation=cv2_interp_codes[interpolation],
            )
        if not return_scale:
            return resized_img
        else:
            new_h, new_w = (
                resized_img.shape[:2]
                if isinstance(resized_img, np.ndarray)
                else resized_img.shape[1:]
            )
            w_scale = new_w / w
            h_scale = new_h / h
            return resized_img, w_scale, h_scale

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""
        if results.get("img", None) is not None:
            if self.keep_ratio:
                img, w_scale, h_scale = self.imresize(
                    results["img"],
                    results["scale"],
                    return_scale=True,
                    interpolation=self.interpolation,
                )
            else:
                img = F.resize(
                    results["img"],
                    results["scale"],
                    interpolation=InterpolationMode.BILINEAR,
                )
                _, h, w = results["img"].size()
                w_scale = img.size()[2] / w
                h_scale = img.size()[1] / h

            results["img"] = img
            results["img_shape"] = (
                img.shape[:2] if isinstance(img, np.ndarray) else img.shape[1:]
            )
            results["scale_factor"] = (w_scale, h_scale)
            results["keep_ratio"] = self.keep_ratio

    def _resize_masks(self, results: dict) -> None:
        """Resize masks with ``results['scale']``"""
        if results.get("gt_masks", None) is not None:
            if self.keep_ratio:
                results["gt_masks"] = results["gt_masks"].rescale(results["scale"])
            else:
                results["gt_masks"] = results["gt_masks"].resize(results["img_shape"])

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get("gt_bboxes", None) is not None:
            results["gt_bboxes"].rescale_(results["scale_factor"])
            if self.clip_object_border:
                results["gt_bboxes"].clip_(results["img_shape"])

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get("gt_seg_map", None) is not None:
            if self.keep_ratio:
                gt_seg = simplecv_imrescale(
                    results["gt_seg_map"],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            else:
                gt_seg = simplecv_imresize(
                    results["gt_seg_map"],
                    results["scale"],
                    interpolation="nearest",
                    backend=self.backend,
                )
            results["gt_seg_map"] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get("gt_keypoints", None) is not None:
            keypoints = results["gt_keypoints"]

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results["scale_factor"]
            )
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(
                    keypoints[:, :, 0], 0, results["img_shape"][1]
                )
                keypoints[:, :, 1] = np.clip(
                    keypoints[:, :, 1], 0, results["img_shape"][0]
                )
            results["gt_keypoints"] = keypoints

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the Resize."""
        w_scale, h_scale = results["scale_factor"]
        homography_matrix = np.array(
            [[w_scale, 0, 0], [0, h_scale, 0], [0, 0, 1]], dtype=np.float32
        )
        if results.get("homography_matrix", None) is None:
            results["homography_matrix"] = homography_matrix
        else:
            results["homography_matrix"] = (
                homography_matrix @ results["homography_matrix"]
            )

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes and semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'scale', 'scale_factor', 'height', 'width', and 'keep_ratio' keys
            are updated in result dict.
        """
        if self.scale:
            results["scale"] = self.scale
        else:
            img_shape = results["img"].shape[:2]
            results["scale"] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        self._record_homography_matrix(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"scale_factor={self.scale_factor}, "
        repr_str += f"keep_ratio={self.keep_ratio}, "
        repr_str += f"clip_object_border={self.clip_object_border}), "
        repr_str += f"backend={self.backend}), "
        repr_str += f"interpolation={self.interpolation})"
        return repr_str


class RandomFlip(BaseTransform):
    """Flip the image & bbox & keypoints & segmentation map. Added or Updated
    keys: flip, flip_direction, img, gt_bboxes, gt_seg_map, and
    gt_keypoints. There are 3 flip modes:

    - ``prob`` is float, ``direction`` is string: the image will be
      ``direction``ly flipped with probability of ``prob`` .
      E.g., ``prob=0.5``, ``direction='horizontal'``,
      then image will be horizontally flipped with probability of 0.5.

    - ``prob`` is float, ``direction`` is list of string: the image will
      be ``direction[i]``ly flipped with probability of
      ``prob/len(direction)``.
      E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
      then image will be horizontally flipped with probability of 0.25,
      vertically with probability of 0.25.

    - ``prob`` is list of float, ``direction`` is list of string:
      given ``len(prob) == len(direction)``, the image will
      be ``direction[i]``ly flipped with probability of ``prob[i]``.
      E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
      'vertical']``, then image will be horizontally flipped with
      probability of 0.3, vertically with probability of 0.5.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Added Keys:

    - flip
    - flip_direction
    - swap_seg_labels (optional)

    Args:
        prob (float | list[float], optional): The flipping probability.
            Defaults to None.
        direction(str | list[str]): The flipping direction. Options
            If input is a list, the length must equal ``prob``. Each
            element in ``prob`` indicates the flip probability of
            corresponding direction. Defaults to 'horizontal'.
        swap_seg_labels (list, optional): The label pair need to be swapped
            for ground truth, like 'left arm' and 'right arm' need to be
            swapped after horizontal flipping. For example, ``[(1, 5)]``,
            where 1/5 is the label of the left/right arm. Defaults to None.
    """

    def __init__(
        self,
        prob: Optional[Union[float, Iterable[float]]] = None,
        direction: Union[str, Sequence[Optional[str]]] = "horizontal",
        swap_seg_labels: Optional[Sequence] = None,
    ) -> None:
        if isinstance(prob, list):
            assert mmengine.is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(
                f"probs must be float or list of float, but \
                              got `{type(prob)}`."
            )
        self.prob = prob
        self.swap_seg_labels = swap_seg_labels

        valid_directions = ["horizontal", "vertical", "diagonal"]
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmengine.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(
                f"direction must be either str or list of str, \
                               but got `{type(direction)}`."
            )
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    def _flip_bbox(
        self, bboxes: np.ndarray, img_shape: Tuple[int, int], direction: str
    ) -> np.ndarray:
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical', and 'diagonal'.

        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        h, w = img_shape
        if direction == "horizontal":
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == "vertical":
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == "diagonal":
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'"
            )
        return flipped

    def _flip_keypoints(
        self,
        keypoints: np.ndarray,
        img_shape: Tuple[int, int],
        direction: str,
    ) -> np.ndarray:
        """Flip keypoints horizontally, vertically or diagonally.

        Args:
            keypoints (numpy.ndarray): Keypoints, shape (..., 2)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical', and 'diagonal'.

        Returns:
            numpy.ndarray: Flipped keypoints.
        """

        meta_info = keypoints[..., 2:]
        keypoints = keypoints[..., :2]
        flipped = keypoints.copy()
        h, w = img_shape
        if direction == "horizontal":
            flipped[..., 0::2] = w - keypoints[..., 0::2]
        elif direction == "vertical":
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        elif direction == "diagonal":
            flipped[..., 0::2] = w - keypoints[..., 0::2]
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'"
            )
        flipped = np.concatenate([flipped, meta_info], axis=-1)
        return flipped

    def _flip_seg_map(self, seg_map: dict, direction: str) -> np.ndarray:
        """Flip segmentation map horizontally, vertically or diagonally.

        Args:
            seg_map (numpy.ndarray): segmentation map, shape (H, W).
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.

        Returns:
            numpy.ndarray: Flipped segmentation map.
        """
        seg_map = simplecv_imflip(seg_map, direction=direction)
        if self.swap_seg_labels is not None:
            # to handle datasets with left/right annotations
            # like 'Left-arm' and 'Right-arm' in LIP dataset
            # Modified from https://github.com/openseg-group/openseg.pytorch/blob/master/lib/datasets/tools/cv2_aug_transforms.py # noqa:E501
            # Licensed under MIT license
            temp = seg_map.copy()
            assert isinstance(self.swap_seg_labels, (tuple, list))
            for pair in self.swap_seg_labels:
                assert isinstance(pair, (tuple, list)) and len(pair) == 2, (
                    "swap_seg_labels must be a sequence with pair, but got "
                    f"{self.swap_seg_labels}."
                )
                seg_map[temp == pair[0]] = pair[1]
                seg_map[temp == pair[1]] = pair[0]
        return seg_map

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction, Sequence) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1.0 - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the RandomFlip."""
        cur_dir = results["flip_direction"]
        h, w = results["img"].shape[:2]

        if cur_dir == "horizontal":
            homography_matrix = np.array(
                [[-1, 0, w], [0, 1, 0], [0, 0, 1]], dtype=np.float32
            )
        elif cur_dir == "vertical":
            homography_matrix = np.array(
                [[1, 0, 0], [0, -1, h], [0, 0, 1]], dtype=np.float32
            )
        elif cur_dir == "diagonal":
            homography_matrix = np.array(
                [[-1, 0, w], [0, -1, h], [0, 0, 1]], dtype=np.float32
            )
        else:
            homography_matrix = np.eye(3, dtype=np.float32)

        if results.get("homography_matrix", None) is None:
            results["homography_matrix"] = homography_matrix
        else:
            results["homography_matrix"] = (
                homography_matrix @ results["homography_matrix"]
            )

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        flip_dims = {"horizontal": [2], "vertical": [1], "diagonal": [1, 2]}
        assert results["flip_direction"] in flip_dims

        if results["torch"]:
            results["img"] = torch.flip(
                results["img"], dims=flip_dims[results["flip_direction"]]
            )
        else:
            results["img"] = np.flip(
                results["img"],
                axis=[i - 1 for i in flip_dims[results["flip_direction"]]],
            )

        img_shape = results["img"].shape[1:]

        # flip bboxes
        if results.get("gt_bboxes", None) is not None:
            results["gt_bboxes"].flip_(img_shape, results["flip_direction"])

        # flip masks
        if results.get("gt_masks", None) is not None:
            results["gt_masks"] = results["gt_masks"].flip(results["flip_direction"])

        # flip segs
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = simplecv_imflip(
                results["gt_seg_map"], direction=results["flip_direction"]
            )

        # record homography matrix for flip
        self._record_homography_matrix(results)

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints."""
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results["flip"] = False
            results["flip_direction"] = None
        else:
            results["flip"] = True
            results["flip_direction"] = cur_dir
            self._flip(results)

    def transform(self, results: dict) -> dict:
        """Transform function to flip images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'flip', and 'flip_direction' keys are
            updated in result dict.
        """
        self._flip_on_direction(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(prob={self.prob}, "
        repr_str += f"direction={self.direction})"

        return repr_str


class Pad(BaseTransform):
    """Pad the image & segmentation map.

    There are three padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number. and (3)pad to square. Also,
    pad to square and pad to the minimum size can be used as the same time.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)

    Modified Keys:

    - img
    - gt_seg_map
    - img_shape

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (w, h). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (Number | dict[str, Number], optional): Padding value for if
            the pad_mode is "constant". If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.

            Defaults to dict(img=0, seg=255).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        size_divisor: Optional[int] = None,
        pad_to_square: bool = False,
        pad_val: Union[Number, dict] = dict(img=0, seg=255),
        padding_mode: str = "constant",
    ) -> None:
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, int):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(pad_val, dict), "pad_val "
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None, (
                "The size and size_divisor must be None " "when pad2square is True"
            )
        else:
            assert (
                size is not None or size_divisor is not None
            ), "only one of size and size_divisor should be valid"
            assert size is None or size_divisor is None
        assert padding_mode in ["constant", "edge", "reflect", "symmetric"]
        self.padding_mode = padding_mode

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get("img", 0)

        size = None
        if self.pad_to_square:
            max_size = max(results["img"].shape[1:])
            size = (max_size, max_size)
        if self.size_divisor is not None:
            if size is None:
                size = (results["img"].size()[1], results["img"].size()[2])
            pad_h = int(np.ceil(size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size[::-1]
        if isinstance(pad_val, int) and results["img"].ndim == 3:
            pad_val = tuple(pad_val for _ in range(results["img"].shape[2]))

        width = max(size[1] - results["img"].shape[2 if results["torch"] else 1], 0)
        height = max(size[0] - results["img"].shape[1 if results["torch"] else 0], 0)
        padding = [0, 0, width, height]

        if not results["torch"]:
            padded_img = np.pad(
                results["img"],
                ((0, height), (0, width), (0, 0)),
                mode="constant",
                constant_values=pad_val[0],
            )
        else:
            padded_img = F.pad(results["img"], padding, pad_val, self.padding_mode)
        # padded_img = simplecv_impad(
        #     results["img"], shape=size, pad_val=pad_val, padding_mode=self.padding_mode
        # )

        results["img"] = padded_img
        results["pad_shape"] = padded_img.shape
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor
        results["img_shape"] = (
            padded_img.shape[1:] if results["torch"] else padded_img.shape[:2]
        )

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        if results.get("gt_seg_map", None) is not None:
            pad_val = self.pad_val.get("seg", 255)
            if isinstance(pad_val, int) and results["gt_seg_map"].ndim == 3:
                pad_val = tuple(pad_val for _ in range(results["gt_seg_map"].shape[2]))
            results["gt_seg_map"] = simplecv_impad(
                results["gt_seg_map"],
                shape=results["pad_shape"][1:],
                pad_val=pad_val,
                padding_mode=self.padding_mode,
            )

    def _pad_masks(self, results: dict) -> None:
        """Pad masks according to ``results['pad_shape']``."""
        if results.get("gt_masks", None) is not None:
            pad_val = self.pad_val.get("masks", 0)
            pad_shape = results["pad_shape"][1:]
            results["gt_masks"] = results["gt_masks"].pad(pad_shape, pad_val=pad_val)

    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)
        self._pad_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_to_square={self.pad_to_square}, "
        repr_str += f"pad_val={self.pad_val}), "
        repr_str += f"padding_mode={self.padding_mode})"
        return repr_str


class RandomCrop(BaseTransform):
    """Random crop the image & bboxes & masks.

    The absolute ``crop_size`` is sampled based on ``crop_type`` and
    ``image_size``, then the cropped results are generated.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)
    - gt_seg_map (optional)
    - gt_instances_ids (options, only used in MOT/VIS)

    Added Keys:

    - homography_matrix

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            (width, height).
        crop_type (str, optional): One of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])].
            Defaults to "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Defaults to False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Defaults to False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          ``gt_bboxes`` corresponds to ``gt_labels`` and ``gt_masks``, and
          ``gt_bboxes_ignore`` corresponds to ``gt_labels_ignore`` and
          ``gt_masks_ignore``.
        - If the crop does not contain any gt-bbox region and
          ``allow_negative_crop`` is set to False, skip this image.
    """

    def __init__(
        self,
        crop_size: tuple,
        crop_type: str = "absolute",
        allow_negative_crop: bool = False,
        recompute_bbox: bool = False,
        bbox_clip_border: bool = True,
    ) -> None:
        if crop_type not in [
            "relative_range",
            "relative",
            "absolute",
            "absolute_range",
        ]:
            raise ValueError(f"Invalid crop_type {crop_type}.")
        if crop_type in ["absolute", "absolute_range"]:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(crop_size[1], int)
            if crop_type == "absolute_range":
                assert crop_size[0] <= crop_size[1]
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox

    def _crop_data(
        self, results: dict, crop_size: Tuple[int, int], allow_negative_crop: bool
    ) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results["img"]
        h, w = img.shape[:2] if isinstance(img, np.ndarray) else img.shape[1:]
        margin_h = max(h - crop_size[0], 0)
        margin_w = max(w - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]], dtype=np.float32
        )

        if results.get("homography_matrix", None) is None:
            results["homography_matrix"] = homography_matrix
        else:
            results["homography_matrix"] = (
                homography_matrix @ results["homography_matrix"]
            )

        # crop the image
        img = (
            img[crop_y1:crop_y2, crop_x1:crop_x2, :]
            if isinstance(img, np.ndarray)
            else img[:, crop_y1:crop_y2, crop_x1:crop_x2]
        )
        img_shape = img.shape
        img_hw = img_shape[:2] if isinstance(img, np.ndarray) else img_shape[1:]
        results["img"] = img
        results["img_shape"] = img_hw

        # crop bboxes accordingly and clip to the image boundary
        if results.get("gt_bboxes", None) is not None:
            bboxes = results["gt_bboxes"]
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_hw)
            valid_inds = bboxes.is_inside(img_hw).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if not valid_inds.any() and not allow_negative_crop:
                return None

            results["gt_bboxes"] = bboxes[valid_inds]

            if results.get("gt_ignore_flags", None) is not None:
                results["gt_ignore_flags"] = results["gt_ignore_flags"][valid_inds]

            if results.get("gt_bboxes_labels", None) is not None:
                results["gt_bboxes_labels"] = results["gt_bboxes_labels"][valid_inds]

            if results.get("gt_masks", None) is not None:
                results["gt_masks"] = results["gt_masks"][valid_inds.nonzero()[0]].crop(
                    np.asarray([crop_x1, crop_y1, crop_x2, crop_y2])
                )
                if self.recompute_bbox:
                    results["gt_bboxes"] = results["gt_masks"].get_bboxes(
                        type(results["gt_bboxes"])
                    )

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get("gt_instances_ids", None) is not None:
                results["gt_instances_ids"] = results["gt_instances_ids"][valid_inds]

        # crop semantic seg
        if results.get("gt_seg_map", None) is not None:
            results["gt_seg_map"] = results["gt_seg_map"][
                crop_y1:crop_y2, crop_x1:crop_x2
            ]

        return results

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return offset_h, offset_w

    @cache_randomness
    def _get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (Tuple[int, int]): (h, w).

        Returns:
            crop_size (Tuple[int, int]): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == "absolute":
            return min(self.crop_size[1], h), min(self.crop_size[0], w)
        elif self.crop_type == "absolute_range":
            crop_h = np.random.randint(
                min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1
            )
            crop_w = np.random.randint(
                min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1
            )
            return crop_h, crop_w
        elif self.crop_type == "relative":
            crop_w, crop_h = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        else:
            # 'relative_range'
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        image_size = (
            results["img"].shape[:2]
            if isinstance(results["img"], np.ndarray)
            else results["img"].shape[1:]
        )
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(crop_size={self.crop_size}, "
        repr_str += f"crop_type={self.crop_type}, "
        repr_str += f"allow_negative_crop={self.allow_negative_crop}, "
        repr_str += f"recompute_bbox={self.recompute_bbox}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


class HSVRandomAug(BaseTransform):
    """Apply HSV augmentation to image sequentially. It is referenced from
    https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py#L21.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delat of value. Defaults to 30.
    """

    def __init__(
        self,
        hue_delta: Union[int, float] = 0.015,
        saturation_delta: Union[int, float] = 0.7,
        value_delta: Union[int, float] = 0.4,
    ):
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta
        self.torch_Aug = tv.transforms.ColorJitter(
            hue=self.hue_delta,
            saturation=self.saturation_delta,
            brightness=self.value_delta,
        )

    def transform(self, results: dict) -> dict:
        """The HSV augmentation transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        if results.get("torch", False):
            results["img"] = self.torch_Aug(results["img"])
        else:
            hsv_gains = (
                random.uniform(-1, 1, 3)
                * [self.hue_delta, self.saturation_delta, self.value_delta]
                + 1
            )
            hue, sat, val = cv2.split(cv2.cvtColor(results["img"], cv2.COLOR_BGR2HSV))

            table_list = np.arange(0, 256, dtype=hsv_gains.dtype)
            lut_hue = ((table_list * hsv_gains[0]) % 180).astype(np.uint8)
            lut_sat = np.clip(table_list * hsv_gains[1], 0, 255).astype(np.uint8)
            lut_val = np.clip(table_list * hsv_gains[2], 0, 255).astype(np.uint8)

            im_hsv = cv2.merge(
                (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
            )
            results["img"] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
            results["torch"] = False
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(hue_delta={self.hue_delta}, "
        repr_str += f"saturation_delta={self.saturation_delta}, "
        repr_str += f"value_delta={self.value_delta})"
        return repr_str


class BaseMixImageTransform(BaseTransform, metaclass=ABCMeta):
    """A Base Transform of multiple images mixed.

    Suitable for training on multiple images mixed data augmentation like
    mosaic and mixup.

    Cached mosaic transform will random select images from the cache
    and combine them into one output image if use_cached is True.

    Args:
        pre_transform(Sequence[str]): Sequence of transform object or
            config dict to be composed. Defaults to None.
        prob(float): The transformation probability. Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(
        self,
        pre_transform: Optional[Sequence[str]] = None,
        prob: float = 1.0,
        use_cached: bool = False,
        max_cached_images: int = 40,
        random_pop: bool = True,
        max_refetch: int = 15,
    ):

        self.max_refetch = max_refetch
        self.prob = prob

        self.use_cached = use_cached
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.results_cache = []

        if pre_transform is None:
            self.pre_transform = None
        else:
            self.pre_transform = Compose(pre_transform)

    @abstractmethod
    def get_indexes(self, dataset: Union[BaseDataset, list]) -> Union[list, int]:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list or int: indexes.
        """
        pass

    @abstractmethod
    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        pass

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Data augmentation function.

        The transform steps are as follows:
        1. Randomly generate index list of other images.
        2. Before Mosaic or MixUp need to go through the necessary
            pre_transform, such as MixUp' pre_transform pipeline
            include: 'LoadImageFromFile','LoadAnnotations',
            'Mosaic' and 'RandomAffine'.
        3. Use mix_img_transform function to implement specific
            mix operations.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """

        if random.uniform(0, 1) > self.prob:
            return results

        if self.use_cached:
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            dataset = results.pop("dataset", None)
            self.results_cache.append(copy.deepcopy(results))
            if len(self.results_cache) > self.max_cached_images:
                if self.random_pop:
                    index = random.randint(0, len(self.results_cache) - 1)
                else:
                    index = 0
                self.results_cache.pop(index)

            if len(self.results_cache) <= 4:
                return results
        else:
            assert "dataset" in results
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            dataset = results.pop("dataset", None)

        for _ in range(self.max_refetch):
            # get index of one or three other images
            if self.use_cached:
                indexes = self.get_indexes(self.results_cache)
            else:
                indexes = self.get_indexes(dataset)

            if not isinstance(indexes, collections.abc.Sequence):
                indexes = [indexes]

            if self.use_cached:
                mix_results = [copy.deepcopy(self.results_cache[i]) for i in indexes]
            else:
                # get images information will be used for Mosaic or MixUp
                mix_results = [
                    copy.deepcopy(dataset.get_data_info(index)) for index in indexes
                ]

            if self.pre_transform is not None:
                for i, data in enumerate(mix_results):
                    # pre_transform may also require dataset
                    data.update({"dataset": dataset})
                    # before Mosaic or MixUp need to go through
                    # the necessary pre_transform
                    _results = self.pre_transform(data)
                    _results.pop("dataset")
                    mix_results[i] = _results

            if None not in mix_results:
                results["mix_results"] = mix_results
                break
            print("Repeated calculation")
        else:
            raise RuntimeError(
                "The loading pipeline of the original dataset"
                " always return None. Please check the correctness "
                "of the dataset and its pipeline."
            )

        # Mosaic or MixUp
        results = self.mix_img_transform(results)

        if "mix_results" in results:
            results.pop("mix_results")
        results["dataset"] = dataset

        return results


class Mosaic(BaseMixImageTransform):
    """Mosaic augmentation.

    Given 4 images, mosaic transform combines them into
    one output image. The output image is composed of the parts from each sub-
    image.

    .. code:: text

                        mosaic transform
                           center_x
                +------------------------------+
                |       pad        |           |
                |      +-----------+    pad    |
                |      |           |           |
                |      |  image1   +-----------+
                |      |           |           |
                |      |           |   image2  |
     center_y   |----+-+-----------+-----------+
                |    |   cropped   |           |
                |pad |   image3    |   image4  |
                |    |             |           |
                +----|-------------+-----------+
                     |             |
                     +-------------+

     The mosaic transform steps are as follows:

         1. Choose the mosaic center as the intersections of 4 images
         2. Get the left top image according to the index, and randomly
            sample another 3 images from the custom dataset.
         3. Sub image will be cropped if image is larger than mosaic patch

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        img_scale (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (width, height).
            Defaults to (640, 640).
        center_ratio_range (Sequence[float]): Center ratio range of mosaic
            output. Defaults to (0.5, 1.5).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pad_val (int): Pad value. Defaults to 114.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 40.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of retry iterations for getting
            valid results from the pipeline. If the number of iterations is
            greater than `max_refetch`, but results is still None, then the
            iteration is terminated and raise the error. Defaults to 15.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        center_ratio_range: Tuple[float, float] = (0.5, 1.5),
        bbox_clip_border: bool = True,
        pad_val: float = 114.0,
        pre_transform: Sequence[dict] = None,
        prob: float = 1.0,
        use_cached: bool = True,
        max_cached_images: int = 40,
        random_pop: bool = True,
        max_refetch: int = 15,
    ):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, (
            "The probability should be in range [0,1]. " f"got {prob}."
        )
        if use_cached:
            assert max_cached_images >= 4, (
                "The length of cache must >= 4, " f"but got {max_cached_images}."
            )

        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch,
        )

        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.bbox_clip_border = bbox_clip_border
        self.pad_val = pad_val

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> list:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            list: indexes.
        """
        indexes = [random.randint(0, len(dataset)) for _ in range(3)]
        return indexes

    def mix_img_transform(self, results: dict) -> dict:
        """Mixed image data transformation.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert "mix_results" in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_masks = []
        mosaic_kps = []
        with_mask = True if "gt_masks" in results else False
        with_kps = True if "gt_keypoints" in results else False
        # self.img_scale is wh format
        img_scale_w, img_scale_h = self.img_scale

        if len(results["img"].shape) == 3:
            if results["torch"]:
                mosaic_img = torch.full(
                    (3, int(img_scale_h * 2), int(img_scale_w * 2)),
                    self.pad_val,
                    dtype=results["img"].dtype,
                )
            else:
                mosaic_img = np.full(
                    (int(img_scale_h * 2), int(img_scale_w * 2), 3),
                    self.pad_val,
                    dtype=results["img"].dtype,
                )

        else:
            mosaic_img = torch.full(
                (int(img_scale_h * 2), int(img_scale_w * 2)),
                self.pad_val,
                dtype=results["img"].dtype,
            )

        center_x = int(random.uniform(*self.center_ratio_range) * img_scale_w)
        center_y = int(random.uniform(*self.center_ratio_range) * img_scale_h)
        center_position = (center_x, center_y)

        loc_strs = ("top_left", "top_right", "bottom_left", "bottom_right")
        for i, loc in enumerate(loc_strs):
            if loc == "top_left":
                results_patch = results
            else:
                results_patch = results["mix_results"][i - 1]

            img_i = results_patch["img"]
            h_i, w_i = img_i.shape[1:] if results["torch"] else img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(img_scale_h / h_i, img_scale_w / w_i)
            if results["torch"]:
                img_i = F.resize(
                    img_i, [int(h_i * scale_ratio_i), int(w_i * scale_ratio_i)]
                )
            else:
                img_i = cv2.resize(
                    img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)), img_i
                )

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc,
                center_position,
                img_i.shape[1:][::-1] if results["torch"] else img_i.shape[:2][::-1],
            )
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            if results["torch"]:
                mosaic_img[:, y1_p:y2_p, x1_p:x2_p] = img_i[:, y1_c:y2_c, x1_c:x2_c]
            else:
                mosaic_img[y1_p:y2_p, x1_p:x2_p, :] = img_i[y1_c:y2_c, x1_c:x2_c, :]

            # adjust coordinate
            gt_bboxes_i = results_patch["gt_bboxes"]
            gt_bboxes_labels_i = results_patch["gt_bboxes_labels"]
            gt_ignore_flags_i = results_patch["gt_ignore_flags"]

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
            if with_mask and results_patch.get("gt_masks", None) is not None:
                gt_masks_i = results_patch["gt_masks"]
                gt_masks_i = gt_masks_i.resize(
                    img_i.shape[1:] if results["torch"] else img_i.shape[:2]
                )
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                    offset=padw,
                    direction="horizontal",
                )
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                    offset=padh,
                    direction="vertical",
                )
                mosaic_masks.append(gt_masks_i)
            if with_kps and results_patch.get("gt_keypoints", None) is not None:
                gt_kps_i = results_patch["gt_keypoints"]
                gt_kps_i.rescale_([scale_ratio_i, scale_ratio_i])
                gt_kps_i.translate_([padw, padh])
                mosaic_kps.append(gt_kps_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * img_scale_h, 2 * img_scale_w])
            if with_mask:
                mosaic_masks = mosaic_masks[0].cat(mosaic_masks)
                results["gt_masks"] = mosaic_masks
            if with_kps:
                mosaic_kps = mosaic_kps[0].cat(mosaic_kps, 0)
                mosaic_kps.clip_([2 * img_scale_h, 2 * img_scale_w])
                results["gt_keypoints"] = mosaic_kps
        else:
            # remove outside bboxes
            inside_inds = mosaic_bboxes.is_inside(
                [2 * img_scale_h, 2 * img_scale_w]
            ).numpy()
            mosaic_bboxes = mosaic_bboxes[inside_inds]
            mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
            mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]
            if with_mask:
                mosaic_masks = mosaic_masks[0].cat(mosaic_masks)[inside_inds]
                results["gt_masks"] = mosaic_masks
            if with_kps:
                mosaic_kps = mosaic_kps[0].cat(mosaic_kps, 0)
                mosaic_kps = mosaic_kps[inside_inds]
                results["gt_keypoints"] = mosaic_kps

        results["img"] = mosaic_img
        results["img_shape"] = (
            mosaic_img.shape[1:] if results["torch"] else mosaic_img.shape[:2]
        )
        results["gt_bboxes"] = mosaic_bboxes
        results["gt_bboxes_labels"] = mosaic_bboxes_labels
        results["gt_ignore_flags"] = mosaic_ignore_flags

        return results

    def _mosaic_combine(
        self, loc: str, center_position_xy: Sequence[float], img_shape_wh: Sequence[int]
    ) -> Tuple[Tuple[int], Tuple[int]]:
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """
        assert loc in ("top_left", "top_right", "bottom_left", "bottom_right")
        if loc == "top_left":
            # index0 to top left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                max(center_position_xy[1] - img_shape_wh[1], 0),
                center_position_xy[0],
                center_position_xy[1],
            )
            crop_coord = (
                img_shape_wh[0] - (x2 - x1),
                img_shape_wh[1] - (y2 - y1),
                img_shape_wh[0],
                img_shape_wh[1],
            )

        elif loc == "top_right":
            # index1 to top right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                max(center_position_xy[1] - img_shape_wh[1], 0),
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[0] * 2),
                center_position_xy[1],
            )
            crop_coord = (
                0,
                img_shape_wh[1] - (y2 - y1),
                min(img_shape_wh[0], x2 - x1),
                img_shape_wh[1],
            )

        elif loc == "bottom_left":
            # index2 to bottom left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                center_position_xy[1],
                center_position_xy[0],
                min(self.img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = (
                img_shape_wh[0] - (x2 - x1),
                0,
                img_shape_wh[0],
                min(y2 - y1, img_shape_wh[1]),
            )

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                center_position_xy[1],
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[0] * 2),
                min(self.img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = (
                0,
                0,
                min(img_shape_wh[0], x2 - x1),
                min(y2 - y1, img_shape_wh[1]),
            )

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"center_ratio_range={self.center_ratio_range}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"prob={self.prob})"
        return repr_str


class MixUp(BaseMixImageTransform):
    """MixUp data augmentation for YOLOX.

    .. code:: text

                         mixup transform
                +---------------+--------------+
                | mixup image   |              |
                |      +--------|--------+     |
                |      |        |        |     |
                +---------------+        |     |
                |      |                 |     |
                |      |      image      |     |
                |      |                 |     |
                |      |                 |     |
                |      +-----------------+     |
                |             pad              |
                +------------------------------+

    The mixup transform steps are as follows:

        1. Another random image is picked by dataset and embedded in
           the top left patch(after padding and resizing)
        2. The target of mixup transform is the weighted average of mixup
           image and origin image.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - mix_results (List[dict])


    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)


    Args:
        img_scale (Sequence[int]): Image output size after mixup pipeline.
            The shape order should be (width, height). Defaults to (640, 640).
        ratio_range (Sequence[float]): Scale ratio of mixup image.
            Defaults to (0.5, 1.5).
        flip_ratio (float): Horizontal flip ratio of mixup image.
            Defaults to 0.5.
        pad_val (int): Pad value. Defaults to 114.
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        pre_transform(Sequence[dict]): Sequence of transform object or
            config dict to be composed.
        prob (float): Probability of applying this transformation.
            Defaults to 1.0.
        use_cached (bool): Whether to use cache. Defaults to False.
        max_cached_images (int): The maximum length of the cache. The larger
            the cache, the stronger the randomness of this transform. As a
            rule of thumb, providing 10 caches for each image suffices for
            randomness. Defaults to 20.
        random_pop (bool): Whether to randomly pop a result from the cache
            when the cache is full. If set to False, use FIFO popping method.
            Defaults to True.
        max_refetch (int): The maximum number of iterations. If the number of
            iterations is greater than `max_refetch`, but gt_bbox is still
            empty, then the iteration is terminated. Defaults to 15.
    """

    def __init__(
        self,
        img_scale: Tuple[int, int] = (640, 640),
        ratio_range: Tuple[float, float] = (0.5, 1.5),
        flip_ratio: float = 0.5,
        pad_val: float = 114.0,
        bbox_clip_border: bool = True,
        pre_transform: Sequence[dict] = None,
        prob: float = 1.0,
        use_cached: bool = True,
        max_cached_images: int = 20,
        random_pop: bool = True,
        max_refetch: int = 15,
    ):
        assert isinstance(img_scale, tuple)
        if use_cached:
            assert max_cached_images >= 2, (
                "The length of cache must >= 2, " f"but got {max_cached_images}."
            )
        super().__init__(
            pre_transform=pre_transform,
            prob=prob,
            use_cached=use_cached,
            max_cached_images=max_cached_images,
            random_pop=random_pop,
            max_refetch=max_refetch,
        )
        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.flip_ratio = flip_ratio
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border

    def get_indexes(self, dataset: Union[BaseDataset, list]) -> int:
        """Call function to collect indexes.

        Args:
            dataset (:obj:`Dataset` or list): The dataset or cached list.

        Returns:
            int: indexes.
        """
        return random.randint(0, len(dataset))

    def mix_img_transform(self, results: dict) -> dict:
        """YOLOX MixUp transform function.

        Args:
            results (dict): Result dict.

        Returns:
            results (dict): Updated result dict.
        """
        assert "mix_results" in results
        assert len(results["mix_results"]) == 1, "MixUp only support 2 images now !"

        if results["mix_results"][0]["gt_bboxes"].shape[0] == 0:
            # empty bbox
            return results

        retrieve_results = results["mix_results"][0]
        retrieve_img = retrieve_results["img"]

        jit_factor = random.uniform(*self.ratio_range)
        is_filp = random.uniform(0, 1) > self.flip_ratio

        if len(retrieve_img.shape) == 3:
            if results["torch"]:
                out_img = torch.full(
                    (3, self.img_scale[1], self.img_scale[0]),
                    self.pad_val,
                    dtype=retrieve_img.dtype,
                )
            else:
                out_img = np.full(
                    (self.img_scale[1], self.img_scale[0], 3),
                    self.pad_val,
                    dtype=retrieve_img.dtype,
                )
            img_shape = (
                retrieve_img.shape[1:] if results["torch"] else retrieve_img.shape[:2]
            )
            out_img_shape = out_img.shape[1:] if results["torch"] else out_img.shape[:2]
        else:
            if results["torch"]:
                out_img = torch.full(
                    (self.img_scale[1], self.img_scale[0]),
                    self.pad_val,
                    dtype=retrieve_img.dtype,
                )
            else:
                out_img = np.full(
                    (self.img_scale[1], self.img_scale[0]),
                    self.pad_val,
                    dtype=retrieve_img.dtype,
                )
            img_shape = retrieve_img.shape
            out_img_shape = out_img.shape

        # 1. keep_ratio resize
        scale_ratio = min(
            self.img_scale[1] / img_shape[0],  # h
            self.img_scale[0] / img_shape[1],  # w
        )

        if results["torch"]:
            retrieve_img = F.resize(
                retrieve_img,
                [
                    int(img_shape[0] * scale_ratio),
                    int(img_shape[1] * scale_ratio),
                ],
            )
        else:
            retrieve_img = cv2.resize(
                retrieve_img,
                [
                    int(img_shape[1] * scale_ratio),
                    int(img_shape[0] * scale_ratio),
                ],
            )
        img_shape = img_shape = (
            retrieve_img.shape[1:] if results["torch"] else retrieve_img.shape[:2]
        )
        # 2. paste
        if results["torch"]:
            out_img[: img_shape[0], : img_shape[1]] = retrieve_img
        else:
            out_img[: img_shape[0], : img_shape[1]] = retrieve_img

        # 3. scale jit
        scale_ratio *= jit_factor

        if results["torch"]:
            out_img = F.resize(
                out_img,
                [
                    int(out_img_shape[0] * jit_factor),
                    int(out_img_shape[1] * jit_factor),
                ],
            )
        else:
            out_img = cv2.resize(
                out_img,
                [
                    int(out_img_shape[1] * jit_factor),
                    int(out_img_shape[0] * jit_factor),
                ],
            )

        # 4. flip ？
        if is_filp:
            if results["torch"]:
                out_img = torch.flip(out_img, [2])
            else:
                out_img = np.flip(out_img, [1])

        # 5. random crop
        out_img_shape = out_img.shape[1:] if results["torch"] else out_img.shape[:2]
        ori_img = results["img"]
        origin_h, origin_w = out_img_shape
        target_h, target_w = (
            ori_img.shape[1:] if results["torch"] else ori_img.shape[:2]
        )
        if results["torch"]:
            padded_img = (
                torch.ones(
                    3,
                    max(origin_h, target_h),
                    max(origin_w, target_w),
                    dtype=torch.uint8,
                )
                * self.pad_val
            )
        else:
            padded_img = (
                np.ones(
                    (max(origin_h, target_h), max(origin_w, target_w), 3),
                    dtype=np.uint8,
                )
                * self.pad_val
            )
        padded_img[:origin_h, :origin_w] = out_img
        padded_img_shape = (
            padded_img.shape[1:] if results["torch"] else padded_img.shape[:2]
        )
        x_offset, y_offset = 0, 0
        if padded_img_shape[0] > target_h:
            y_offset = random.randint(0, padded_img_shape[0] - target_h)
        if padded_img_shape[1] > target_w:
            x_offset = random.randint(0, padded_img_shape[1] - target_w)
        padded_cropped_img = padded_img[
            y_offset : y_offset + target_h, x_offset : x_offset + target_w
        ]

        # 6. adjust bbox
        retrieve_gt_bboxes = retrieve_results["gt_bboxes"]
        retrieve_gt_bboxes.rescale_([scale_ratio, scale_ratio])
        if self.bbox_clip_border:
            retrieve_gt_bboxes.clip_([origin_h, origin_w])

        if is_filp:
            retrieve_gt_bboxes.flip_([origin_h, origin_w], direction="horizontal")

        # 7. filter
        cp_retrieve_gt_bboxes = retrieve_gt_bboxes.clone()
        cp_retrieve_gt_bboxes.translate_([-x_offset, -y_offset])
        if self.bbox_clip_border:
            cp_retrieve_gt_bboxes.clip_([target_h, target_w])

        # 8. mix up
        mixup_img = 0.5 * ori_img + 0.5 * padded_cropped_img

        retrieve_gt_bboxes_labels = retrieve_results["gt_bboxes_labels"]
        retrieve_gt_ignore_flags = retrieve_results["gt_ignore_flags"]

        mixup_gt_bboxes = cp_retrieve_gt_bboxes.cat(
            (results["gt_bboxes"], cp_retrieve_gt_bboxes), dim=0
        )
        mixup_gt_bboxes_labels = np.concatenate(
            (results["gt_bboxes_labels"], retrieve_gt_bboxes_labels), axis=0
        )
        mixup_gt_ignore_flags = np.concatenate(
            (results["gt_ignore_flags"], retrieve_gt_ignore_flags), axis=0
        )

        if not self.bbox_clip_border:
            # remove outside bbox
            inside_inds = mixup_gt_bboxes.is_inside([target_h, target_w]).numpy()
            mixup_gt_bboxes = mixup_gt_bboxes[inside_inds]
            mixup_gt_bboxes_labels = mixup_gt_bboxes_labels[inside_inds]
            mixup_gt_ignore_flags = mixup_gt_ignore_flags[inside_inds]

        if "gt_keypoints" in results:
            # adjust kps
            retrieve_gt_keypoints = retrieve_results["gt_keypoints"]
            retrieve_gt_keypoints.rescale_([scale_ratio, scale_ratio])
            if self.bbox_clip_border:
                retrieve_gt_keypoints.clip_([origin_h, origin_w])

            if is_filp:
                retrieve_gt_keypoints.flip_(
                    [origin_h, origin_w], direction="horizontal"
                )

            # filter
            cp_retrieve_gt_keypoints = retrieve_gt_keypoints.clone()
            cp_retrieve_gt_keypoints.translate_([-x_offset, -y_offset])
            if self.bbox_clip_border:
                cp_retrieve_gt_keypoints.clip_([target_h, target_w])

            # mixup
            mixup_gt_keypoints = cp_retrieve_gt_keypoints.cat(
                (results["gt_keypoints"], cp_retrieve_gt_keypoints), dim=0
            )
            if not self.bbox_clip_border:
                # remove outside bbox
                mixup_gt_keypoints = mixup_gt_keypoints[inside_inds]
            results["gt_keypoints"] = mixup_gt_keypoints

        results["img"] = mixup_img
        results["img_shape"] = (
            mixup_img.shape[1:] if results["torch"] else mixup_img.shape[:2]
        )
        results["gt_bboxes"] = mixup_gt_bboxes
        results["gt_bboxes_labels"] = mixup_gt_bboxes_labels
        results["gt_ignore_flags"] = mixup_gt_ignore_flags

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(img_scale={self.img_scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"flip_ratio={self.flip_ratio}, "
        repr_str += f"pad_val={self.pad_val}, "
        repr_str += f"max_refetch={self.max_refetch}, "
        repr_str += f"bbox_clip_border={self.bbox_clip_border})"
        return repr_str


class Bbox2FomoMask(BaseTransform):
    def __init__(
        self,
        downsample_factor: Tuple[int, ...] = (8,),
        num_classes: int = 80,
    ) -> None:
        super().__init__()
        self.downsample_factor = downsample_factor
        self.num_classes = num_classes

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        H, W = results["scale"]
        bbox = results["gt_bboxes"]
        labels = results["gt_bboxes_labels"]

        res = []
        for factor in self.downsample_factor:
            Dh, Dw = int(H / factor), int(W / factor)
            target = self.build_target(
                bbox, feature_shape=(Dh, Dw), ori_shape=(W, H), labels=labels
            )
            res.append(target)

        results["fomo_mask"] = copy.deepcopy(res)
        return results

    def build_target(self, bboxs, feature_shape, ori_shape, labels):
        (H, W) = feature_shape
        # target_data = torch.zeros(size=(1,H, W, self.num_classes + 1))
        target_data = np.zeros((1, H, W, self.num_classes + 1))
        target_data[..., 0] = 1
        for idx, i in enumerate(bboxs):
            w = int(i.centers[0][0] / ori_shape[0] * W)
            h = int(i.centers[0][1] / ori_shape[1] * H)
            target_data[0, h-1, w-1, 0] = 0  # background
            target_data[0, h-1, w-1, int(labels[idx] + 1)] = 1  # label
        return target_data
