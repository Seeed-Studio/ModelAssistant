# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings
from typing import Union, Sequence, Tuple, Dict

import numpy as np
from mmengine.utils import is_list_of, is_tuple_of, is_seq_of
from mmengine.registry import TRANSFORMS
from sscma.utils import simplecv_imresize, simplecv_imcrop
from .basetransform import BaseTransform
from .utils import cache_randomness


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
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:

        self.scale = scale
        self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})

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

        assert is_list_of(scales, tuple) and len(scales) == 2
        scale_0 = [scales[0][0], scales[1][0]]
        scale_1 = [scales[0][1], scales[1][1]]
        edge_0 = np.random.randint(min(scale_0), max(scale_0) + 1)
        edge_1 = np.random.randint(min(scale_1), max(scale_1) + 1)
        scale = (edge_0, edge_1)
        return scale

    @staticmethod
    def _random_sample_ratio(scale: tuple, ratio_range: Tuple[float,
                                                              float]) -> tuple:
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
                self.ratio_range)
        elif is_seq_of(self.scale, tuple):
            scale = self._random_sample(self.scale)  # type: ignore
        else:
            raise NotImplementedError('Do not support sampling function '
                                      f'for "{self.scale}"')

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
        results['scale'] = self._random_scale()
        self.resize.scale = results['scale']
        results = self.resize(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'resize_cfg={self.resize_cfg})'
        return repr_str
