# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union, Sequence, Tuple, List, Optional, Dict, Iterable

import numpy as np
from mmengine.utils import is_list_of, is_tuple_of, is_seq_of
from mmengine.registry import TRANSFORMS

from .basetransform import BaseTransform

from sscma.utils import simplecv_imresize, simplecv_imflip, simplecv_imcrop
from sscma.structures.bbox import autocast_box_type

import copy
import functools
import inspect
import weakref
import math
import warnings


class cache_randomness:
    """Decorator that marks the method with random return value(s) in a
    transform class.

    This decorator is usually used together with the context-manager
    :func`:cache_random_params`. In this context, a decorated method will
    cache its return value(s) at the first time of being invoked, and always
    return the cached values when being invoked again.

    .. note::
        Only an instance method can be decorated with ``cache_randomness``.
    """

    def __init__(self, func):
        # Check `func` is to be bound as an instance method
        if not inspect.isfunction(func):
            raise TypeError("Unsupport callable to decorate with" "@cache_randomness.")
        func_args = inspect.getfullargspec(func).args
        if len(func_args) == 0 or func_args[0] != "self":
            raise TypeError(
                "@cache_randomness should only be used to decorate "
                "instance methods (the first argument is ``self``)."
            )

        functools.update_wrapper(self, func)
        self.func = func
        self.instance_ref = None

    def __set_name__(self, owner, name):
        # Maintain a record of decorated methods in the class
        if not hasattr(owner, "_methods_with_randomness"):
            setattr(owner, "_methods_with_randomness", [])

        # Here `name` equals to `self.__name__`, i.e., the name of the
        # decorated function, due to the invocation of `update_wrapper` in
        # `self.__init__()`
        owner._methods_with_randomness.append(name)

    def __call__(self, *args, **kwargs):
        # Get the transform instance whose method is decorated
        # by cache_randomness
        instance = self.instance_ref()
        name = self.__name__

        # Check the flag ``self._cache_enabled``, which should be
        # set by the contextmanagers like ``cache_random_parameters```
        cache_enabled = getattr(instance, "_cache_enabled", False)

        if cache_enabled:
            # Initialize the cache of the transform instances. The flag
            # ``cache_enabled``` is set by contextmanagers like
            # ``cache_random_params```.
            if not hasattr(instance, "_cache"):
                setattr(instance, "_cache", {})

            if name not in instance._cache:
                instance._cache[name] = self.func(instance, *args, **kwargs)
            # Return the cached value
            return instance._cache[name]
        else:
            # Clear cache
            if hasattr(instance, "_cache"):
                del instance._cache
            # Return function output
            return self.func(instance, *args, **kwargs)

    def __get__(self, obj, cls):
        self.instance_ref = weakref.ref(obj)
        # Return a copy to avoid multiple transform instances sharing
        # one `cache_randomness` instance, which may cause data races
        # in multithreading cases.
        return copy.copy(self)


class RandomResizedCrop(BaseTransform):
    """Crop the given image to random scale and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    Args:
        scale (sequence | int): Desired output scale of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        crop_ratio_range (tuple): Range of the random size of the cropped
            image compared to the original image. Defaults to (0.08, 1.0).
        aspect_ratio_range (tuple): Range of the random aspect ratio of the
            cropped image compared to the original image.
            Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maximum number of attempts before falling back to
            Central Crop. Defaults to 10.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accepted values are
            'cv2' and 'pillow'. Defaults to 'cv2'.
    """

    def __init__(
        self,
        scale: Union[Sequence, int],
        crop_ratio_range: Tuple[float, float] = (0.08, 1.0),
        aspect_ratio_range: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        max_attempts: int = 10,
        interpolation: str = "bilinear",
        backend: str = "cv2",
    ) -> None:
        if isinstance(scale, Sequence):
            assert len(scale) == 2
            assert scale[0] > 0 and scale[1] > 0
            self.scale = scale
        else:
            assert scale > 0
            self.scale = (scale, scale)
        if (crop_ratio_range[0] > crop_ratio_range[1]) or (
            aspect_ratio_range[0] > aspect_ratio_range[1]
        ):
            raise ValueError(
                "range should be of kind (min, max). "
                f"But received crop_ratio_range {crop_ratio_range} "
                f"and aspect_ratio_range {aspect_ratio_range}."
            )
        assert (
            isinstance(max_attempts, int) and max_attempts >= 0
        ), "max_attempts mush be int and no less than 0."
        assert interpolation in ("nearest", "bilinear", "bicubic", "area", "lanczos")

        self.crop_ratio_range = crop_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
        self.max_attempts = max_attempts
        self.interpolation = interpolation
        self.backend = backend

    @cache_randomness
    def rand_crop_params(self, img: np.ndarray) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (ndarray): Image to be cropped.

        Returns:
            tuple: Params (offset_h, offset_w, target_h, target_w) to be
                passed to `crop` for a random sized crop.
        """
        h, w = img.shape[:2]
        area = h * w

        for _ in range(self.max_attempts):
            target_area = np.random.uniform(*self.crop_ratio_range) * area
            log_ratio = (
                math.log(self.aspect_ratio_range[0]),
                math.log(self.aspect_ratio_range[1]),
            )
            aspect_ratio = math.exp(np.random.uniform(*log_ratio))
            target_w = int(round(math.sqrt(target_area * aspect_ratio)))
            target_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_w <= w and 0 < target_h <= h:
                offset_h = np.random.randint(0, h - target_h + 1)
                offset_w = np.random.randint(0, w - target_w + 1)

                return offset_h, offset_w, target_h, target_w

        # Fallback to central crop
        in_ratio = float(w) / float(h)
        if in_ratio < min(self.aspect_ratio_range):
            target_w = w
            target_h = int(round(target_w / min(self.aspect_ratio_range)))
        elif in_ratio > max(self.aspect_ratio_range):
            target_h = h
            target_w = int(round(target_h * max(self.aspect_ratio_range)))
        else:  # whole image
            target_w = w
            target_h = h
        offset_h = (h - target_h) // 2
        offset_w = (w - target_w) // 2
        return offset_h, offset_w, target_h, target_w

    def transform(self, results: dict) -> dict:
        """Transform function to randomly resized crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly resized cropped results, 'img_shape'
                key in result dict is updated according to crop size.
        """
        img = results["img"]
        offset_h, offset_w, target_h, target_w = self.rand_crop_params(img)
        img = simplecv_imcrop(
            img,
            bboxes=np.array(
                [offset_w, offset_h, offset_w + target_w - 1, offset_h + target_h - 1]
            ),
        )
        img = simplecv_imresize(
            img,
            tuple(self.scale[::-1]),
            interpolation=self.interpolation,
            backend=self.backend,
        )
        results["img"] = img
        results["img_shape"] = img.shape

        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__ + f"(scale={self.scale}"
        repr_str += ", crop_ratio_range="
        repr_str += f"{tuple(round(s, 4) for s in self.crop_ratio_range)}"
        repr_str += ", aspect_ratio_range="
        repr_str += f"{tuple(round(r, 4) for r in self.aspect_ratio_range)}"
        repr_str += f", max_attempts={self.max_attempts}"
        repr_str += f", interpolation={self.interpolation}"
        repr_str += f", backend={self.backend})"
        return repr_str


class ResizeEdge(BaseTransform):
    """Resize images along the specified edge.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img
    - img_shape

    **Added Keys:**

    - scale
    - scale_factor

    Args:
        scale (int): The edge scale to resizing.
        edge (str): The edge to resize. Defaults to 'short'.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results.
            Defaults to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
            Defaults to 'bilinear'.
    """

    def __init__(
        self,
        scale: int,
        edge: str = "short",
        backend: str = "cv2",
        interpolation: str = "bilinear",
    ) -> None:
        allow_edges = ["short", "long", "width", "height"]
        assert (
            edge in allow_edges
        ), f'Invalid edge "{edge}", please specify from {allow_edges}.'
        self.edge = edge
        self.scale = scale
        self.backend = backend
        self.interpolation = interpolation

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        img, w_scale, h_scale = simplecv_imresize(
            results["img"],
            results["scale"],
            interpolation=self.interpolation,
            return_scale=True,
            backend=self.backend,
        )
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["scale"] = img.shape[:2][::-1]
        results["scale_factor"] = (w_scale, h_scale)

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img', 'scale', 'scale_factor',
            'img_shape' keys are updated in result dict.
        """
        assert "img" in results, "No `img` field in the input."

        h, w = results["img"].shape[:2]
        if any(
            [
                # conditions to resize the width
                self.edge == "short" and w < h,
                self.edge == "long" and w > h,
                self.edge == "width",
            ]
        ):
            width = self.scale
            height = int(self.scale * h / w)
        else:
            height = self.scale
            width = int(self.scale * w / h)
        results["scale"] = (width, height)

        self._resize_img(results)
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"edge={self.edge}, "
        repr_str += f"backend={self.backend}, "
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
            assert is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f"probs must be float or list of float, but \
                              got `{type(prob)}`.")
        self.prob = prob
        self.swap_seg_labels = swap_seg_labels

        valid_directions = ["horizontal", "vertical", "diagonal"]
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f"direction must be either str or list of str, \
                               but got `{type(direction)}`.")
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

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results["img"] = simplecv_imflip(
            results["img"], direction=results["flip_direction"]
        )

        img_shape = results["img"].shape[:2]

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


class CenterCrop(BaseTransform):
    """Crop the center of the image, segmentation masks, bounding boxes and key
    points. If the crop area exceeds the original image and ``auto_pad`` is
    True, the original image will be padded before cropping.

    Required Keys:

    - img
    - gt_seg_map (optional)
    - gt_bboxes (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map (optional)
    - gt_bboxes (optional)
    - gt_keypoints (optional)

    Added Key:

    - pad_shape


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (w, h). If set to an integer, then cropping
            width and height are equal to this integer.
        auto_pad (bool): Whether to pad the image if it's smaller than the
            ``crop_size``. Defaults to False.
        pad_cfg (dict): Base config for padding. Refer to ``mmcv.Pad`` for
            detail. Defaults to ``dict(type='Pad')``.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the
            gt bboxes are allowed to cross the border of images. Therefore,
            we don't need to clip the gt bboxes in these cases.
            Defaults to True.
    """

    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]],
        auto_pad: bool = False,
        pad_cfg: dict = dict(type="Pad"),
        clip_object_border: bool = True,
    ) -> None:
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), "The expected crop_size is an integer, or a tuple containing two "
        "intergers"

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.auto_pad = auto_pad

        self.pad_cfg = pad_cfg.copy()
        # size will be overwritten
        if "size" in self.pad_cfg and auto_pad:
            warnings.warn(
                "``size`` is set in ``pad_cfg``,"
                "however this argument will be overwritten"
                " according to crop size and image size"
            )

        self.clip_object_border = clip_object_border

    def _crop_img(self, results: dict, bboxes: np.ndarray) -> None:
        """Crop image.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if results.get("img", None) is not None:
            img = simplecv_imcrop(results["img"], bboxes=bboxes)
            img_shape = img.shape[:2]  # type: ignore
            results["img"] = img
            results["img_shape"] = img_shape
            results["pad_shape"] = img_shape

    def _crop_seg_map(self, results: dict, bboxes: np.ndarray) -> None:
        """Crop semantic segmentation map.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if results.get("gt_seg_map", None) is not None:
            img = simplecv_imcrop(results["gt_seg_map"], bboxes=bboxes)
            results["gt_seg_map"] = img

    def _crop_bboxes(self, results: dict, bboxes: np.ndarray) -> None:
        """Update bounding boxes according to CenterCrop.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if "gt_bboxes" in results:
            offset_w = bboxes[0]
            offset_h = bboxes[1]
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h])
            # gt_bboxes has shape (num_gts, 4) in (tl_x, tl_y, br_x, br_y)
            # order.
            gt_bboxes = results["gt_bboxes"] - bbox_offset
            if self.clip_object_border:
                gt_bboxes[:, 0::2] = np.clip(
                    gt_bboxes[:, 0::2], 0, results["img"].shape[1]
                )
                gt_bboxes[:, 1::2] = np.clip(
                    gt_bboxes[:, 1::2], 0, results["img"].shape[0]
                )
            results["gt_bboxes"] = gt_bboxes

    def _crop_keypoints(self, results: dict, bboxes: np.ndarray) -> None:
        """Update key points according to CenterCrop. Keypoints that not in the
        cropped image will be set invisible.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if "gt_keypoints" in results:
            offset_w = bboxes[0]
            offset_h = bboxes[1]
            keypoints_offset = np.array([offset_w, offset_h, 0])
            # gt_keypoints has shape (N, NK, 3) in (x, y, visibility) order,
            # NK = number of points per object
            gt_keypoints = results["gt_keypoints"] - keypoints_offset
            # set gt_kepoints out of the result image invisible
            height, width = results["img"].shape[:2]
            valid_pos = (
                (gt_keypoints[:, :, 0] >= 0)
                * (gt_keypoints[:, :, 0] < width)
                * (gt_keypoints[:, :, 1] >= 0)
                * (gt_keypoints[:, :, 1] < height)
            )
            gt_keypoints[:, :, 2] = np.where(valid_pos, gt_keypoints[:, :, 2], 0)
            gt_keypoints[:, :, 0] = np.clip(
                gt_keypoints[:, :, 0], 0, results["img"].shape[1]
            )
            gt_keypoints[:, :, 1] = np.clip(
                gt_keypoints[:, :, 1], 0, results["img"].shape[0]
            )
            results["gt_keypoints"] = gt_keypoints

    def transform(self, results: dict) -> dict:
        """Apply center crop on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Results with CenterCropped image and semantic segmentation
            map.
        """
        crop_width, crop_height = self.crop_size[0], self.crop_size[1]

        assert "img" in results, "`img` is not found in results"
        img = results["img"]
        # img.shape has length 2 for grayscale, length 3 for color
        img_height, img_width = img.shape[:2]

        if crop_height > img_height or crop_width > img_width:
            if self.auto_pad:
                # pad the area
                img_height = max(img_height, crop_height)
                img_width = max(img_width, crop_width)
                pad_size = (img_width, img_height)
                _pad_cfg = self.pad_cfg.copy()
                _pad_cfg.update(dict(size=pad_size))
                pad_transform = TRANSFORMS.build(_pad_cfg)
                results = pad_transform(results)
            else:
                crop_height = min(crop_height, img_height)
                crop_width = min(crop_width, img_width)

        y1 = max(0, int(round((img_height - crop_height) / 2.0)))
        x1 = max(0, int(round((img_width - crop_width) / 2.0)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1
        bboxes = np.array([x1, y1, x2, y2])

        # crop the image
        self._crop_img(results, bboxes)
        # crop the gt_seg_map
        self._crop_seg_map(results, bboxes)
        # crop the bounding box
        self._crop_bboxes(results, bboxes)
        # crop the keypoints
        self._crop_keypoints(results, bboxes)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(crop_size = {self.crop_size}"
        repr_str += f", auto_pad={self.auto_pad}"
        repr_str += f", pad_cfg={self.pad_cfg}"
        repr_str += f",clip_object_border = {self.clip_object_border})"
        return repr_str


class RandomResize(BaseTransform):
    """Random resize images & bbox & keypoints.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if ``scale`` is a sequence of tuple

    .. math::
        target\\_scale[0] \\sim Uniform([scale[0][0], scale[1][0]])
    .. math::
        target\\_scale[1] \\sim Uniform([scale[0][1], scale[1][1]])

    Following the resize order of weight and height in cv2, ``scale[i][0]``
    is for width, and ``scale[i][1]`` is for height.

    - if ``scale`` is a tuple

    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[0]
    .. math::
        target\\_scale[1] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[1]

    Following the resize order of weight and height in cv2, ``ratio_range[0]``
    is for width, and ``ratio_range[1]`` is for height.

    - if ``keep_ratio`` is True, the minimum value of ``target_scale`` will be
      used to set the shorter side and the maximum value will be used to
      set the longer side.

    - if ``keep_ratio`` is False, the value of ``target_scale`` will be used to
      reisze the width and height accordingly.

    Required Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints

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
        scale (tuple or Sequence[tuple]): Images scales for resizing.
            Defaults to None.
        ratio_range (tuple[float], optional): (min_ratio, max_ratio).
            Defaults to None.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
        self,
        scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
        ratio_range: Tuple[float, float] = None,
        resize_type: str = "Resize",
        **resize_kwargs,
    ) -> None:
        self.scale = scale
        self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({"scale": 0, **self.resize_cfg})

    @staticmethod
    def _random_sample(scales: Sequence[Tuple[int, int]]) -> tuple:
        """Private function to randomly sample a scale from a list of tuples.

        Args:
            scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in scales, which specify the lower
                and upper bound of image scales.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert mmengine.is_list_of(scales, tuple) and len(scales) == 2
        scale_0 = [scales[0][0], scales[1][0]]
        scale_1 = [scales[0][1], scales[1][1]]
        edge_0 = np.random.randint(min(scale_0), max(scale_0) + 1)
        edge_1 = np.random.randint(min(scale_1), max(scale_1) + 1)
        scale = (edge_0, edge_1)
        return scale

    @staticmethod
    def _random_sample_ratio(scale: tuple, ratio_range: Tuple[float, float]) -> tuple:
        """Private function to randomly sample a scale from a tuple.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``scale`` to
        generate sampled scale.

        Args:
            scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert isinstance(scale, tuple) and len(scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        return scale

    @cache_randomness
    def _random_scale(self) -> tuple:
        """Private function to randomly sample an scale according to the type
        of ``scale``.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        if is_tuple_of(self.scale, int):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale = self._random_sample_ratio(
                self.scale,  # type: ignore
                self.ratio_range,
            )
        elif is_seq_of(self.scale, tuple):
            scale = self._random_sample(self.scale)  # type: ignore
        else:
            raise NotImplementedError(
                "Do not support sampling function " f'for "{self.scale}"'
            )

        return scale

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, ``img``, ``gt_bboxes``, ``gt_semantic_seg``,
            ``gt_keypoints``, ``scale``, ``scale_factor``, ``img_shape``, and
            ``keep_ratio`` keys are updated in result dict.
        """
        results["scale"] = self._random_scale()
        self.resize.scale = results["scale"]
        results = self.resize(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(scale={self.scale}, "
        repr_str += f"ratio_range={self.ratio_range}, "
        repr_str += f"resize_cfg={self.resize_cfg})"
        return repr_str
