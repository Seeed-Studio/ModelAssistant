# Copyright (c) Seeed Tech Ltd.
# Copyright (c) OpenMMLab.
import collections
import copy
import math
from abc import ABCMeta, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import cv2
import mmcv
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.datasets.transforms import LoadAnnotations as MMDET_LoadAnnotations
from mmdet.datasets.transforms import Resize as MMDET_Resize
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type, get_box_type
from mmdet.structures.mask import PolygonMasks
from mmengine.dataset import BaseDataset
from mmengine.dataset.base_dataset import Compose
from numpy import random

from sscma.registry import TRANSFORMS


@TRANSFORMS.register_module()
class YOLOv5KeepRatioResize(MMDET_Resize):
    """Resize images & bbox(if existed).

    This transform resizes the input image according to ``scale``.
    Bboxes (if existed) are then resized with the same scale factor.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)
    - scale (float)

    Added Keys:

    - scale_factor (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
    """

    def __init__(self, scale: Union[int, Tuple[int, int]], keep_ratio: bool = True, **kwargs):
        assert keep_ratio is True
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

    @staticmethod
    def _get_rescale_ratio(old_size: Tuple[int, int], scale: Union[float, Tuple[int]]) -> float:
        """Calculate the ratio for rescaling.

        Args:
            old_size (tuple[int]): The old size (w, h) of image.
            scale (float | tuple[int]): The scaling factor or maximum size.
                If it is a float number, then the image will be rescaled by
                this factor, else if it is a tuple of 2 integers, then
                the image will be rescaled as large as possible within
                the scale.

        Returns:
            float: The resize ratio.
        """
        w, h = old_size
        if isinstance(scale, (float, int)):
            if scale <= 0:
                raise ValueError(f'Invalid scale {scale}, must be positive.')
            scale_factor = scale
        elif isinstance(scale, tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            scale_factor = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        else:
            raise TypeError('Scale must be a number or tuple of int, ' f'but got {type(scale)}')

        return scale_factor

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        assert self.keep_ratio is True

        if results.get('img', None) is not None:
            image = results['img']
            original_h, original_w = image.shape[:2]
            ratio = self._get_rescale_ratio((original_h, original_w), self.scale)

            if ratio != 1:
                # resize image according to the ratio
                image = mmcv.imrescale(
                    img=image, scale=ratio, interpolation='area' if ratio < 1 else 'bilinear', backend=self.backend
                )

            resized_h, resized_w = image.shape[:2]
            scale_ratio = resized_h / original_h

            scale_factor = (scale_ratio, scale_ratio)

            results['img'] = image
            results['img_shape'] = image.shape[:2]
            results['scale_factor'] = scale_factor


@TRANSFORMS.register_module()
class LetterResize(MMDET_Resize):
    """Resize and pad image while meeting stride-multiple constraints.

    Required Keys:

    - img (np.uint8)
    - batch_shape (np.int64) (optional)

    Modified Keys:

    - img (np.uint8)
    - img_shape (tuple)
    - gt_bboxes (optional)

    Added Keys:
    - pad_param (np.float32)

    Args:
        scale (Union[int, Tuple[int, int]]): Images scales for resizing.
        pad_val (dict): Padding value. Defaults to dict(img=0, seg=255).
        use_mini_pad (bool): Whether using minimum rectangle padding.
            Defaults to True
        stretch_only (bool): Whether stretch to the specified size directly.
            Defaults to False
        allow_scale_up (bool): Allow scale up when ratio > 1. Defaults to True
    """

    def __init__(
        self,
        scale: Union[int, Tuple[int, int]],
        pad_val: dict = dict(img=0, mask=0, seg=255),
        use_mini_pad: bool = False,
        stretch_only: bool = False,
        allow_scale_up: bool = True,
        **kwargs,
    ):
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(pad_val, dict), f'pad_val must be dict, but got {type(pad_val)}'

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up

    def _resize_img(self, results: dict):
        """Resize images with ``results['scale']``."""
        image = results.get('img', None)
        if image is None:
            return

        # Use batch_shape if a batch_shape policy is configured
        if 'batch_shape' in results:
            scale = tuple(results['batch_shape'])  # hw
        else:
            scale = self.scale[::-1]  # wh -> hw
        # scale = self.scale[::-1]

        image_shape = image.shape[:2]  # height, width

        # Scale ratio (new / old)
        ratio = min(scale[0] / image_shape[0], scale[1] / image_shape[1])

        # only scale down, do not scale up (for better test mAP)
        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]  # float -> (float, float) for (height, width)

        # compute the best size of the image
        no_pad_shape = (int(round(image_shape[0] * ratio[0])), int(round(image_shape[1] * ratio[1])))

        # padding height & width
        padding_h, padding_w = [scale[0] - no_pad_shape[0], scale[1] - no_pad_shape[1]]
        if self.use_mini_pad:
            # minimum rectangle padding
            padding_w, padding_h = np.mod(padding_w, 32), np.mod(padding_h, 32)

        elif self.stretch_only:
            # stretch to the specified size directly
            padding_h, padding_w = 0.0, 0.0
            no_pad_shape = (scale[0], scale[1])
            ratio = [scale[0] / image_shape[0], scale[1] / image_shape[1]]  # height, width ratios

        if image_shape != no_pad_shape:
            # compare with no resize and padding size
            image = mmcv.imresize(
                image, (no_pad_shape[1], no_pad_shape[0]), interpolation=self.interpolation, backend=self.backend
            )

        scale_factor = (ratio[1], ratio[0])  # mmcv scale factor is (w, h)

        if 'scale_factor' in results:
            results['scale_factor_origin'] = results['scale_factor']
        results['scale_factor'] = scale_factor

        # padding
        top_padding, left_padding = int(round(padding_h // 2 - 0.1)), int(round(padding_w // 2 - 0.1))
        bottom_padding = padding_h - top_padding
        right_padding = padding_w - left_padding

        padding_list = [top_padding, bottom_padding, left_padding, right_padding]
        if top_padding != 0 or bottom_padding != 0 or left_padding != 0 or right_padding != 0:
            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and image.ndim == 3:
                pad_val = tuple(pad_val for _ in range(image.shape[2]))

            image = mmcv.impad(
                img=image,
                padding=(padding_list[2], padding_list[0], padding_list[3], padding_list[1]),
                pad_val=pad_val,
                padding_mode='constant',
            )
        results['img'] = image
        results['img_shape'] = image.shape
        if 'pad_param' in results:
            results['pad_param_origin'] = results['pad_param'] * np.repeat(ratio, 2)
        results['pad_param'] = np.array(padding_list, dtype=np.float32)

    def _resize_masks(self, results: dict):
        """Resize masks with ``results['scale']``"""
        if results.get('gt_masks', None) is None:
            return

        gt_masks = results['gt_masks']
        assert isinstance(gt_masks, PolygonMasks), f'Only supports PolygonMasks, but got {type(gt_masks)}'

        # resize the gt_masks
        gt_mask_h = results['gt_masks'].height * results['scale_factor'][1]
        gt_mask_w = results['gt_masks'].width * results['scale_factor'][0]
        gt_masks = results['gt_masks'].resize((int(round(gt_mask_h)), int(round(gt_mask_w))))

        top_padding, _, left_padding, _ = results['pad_param']
        if int(left_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results['img_shape'][:2], offset=int(left_padding), direction='horizontal'
            )
        if int(top_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=results['img_shape'][:2], offset=int(top_padding), direction='vertical'
            )
        results['gt_masks'] = gt_masks

    def _resize_bboxes(self, results: dict):
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is None:
            return
        results['gt_bboxes'].rescale_(results['scale_factor'])

        if len(results['pad_param']) != 4:
            return
        results['gt_bboxes'].translate_((results['pad_param'][2], results['pad_param'][0]))

        if self.clip_object_border:
            results['gt_bboxes'].clip_(results['img_shape'])

    def transform(self, results: dict) -> dict:
        results = super().transform(results)
        if 'scale_factor_origin' in results:
            scale_factor_origin = results.pop('scale_factor_origin')
            results['scale_factor'] = (
                results['scale_factor'][0] * scale_factor_origin[0],
                results['scale_factor'][1] * scale_factor_origin[1],
            )
        if 'pad_param_origin' in results:
            pad_param_origin = results.pop('pad_param_origin')
            results['pad_param'] += pad_param_origin
        return results


@TRANSFORMS.register_module()
class YOLOv5HSVRandomAug(BaseTransform):
    """Apply HSV augmentation to image sequentially.

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_delta ([int, float]): delta of hue. Defaults to 0.015.
        saturation_delta ([int, float]): delta of saturation. Defaults to 0.7.
        value_delta ([int, float]): delta of value. Defaults to 0.4.
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

    def transform(self, results: dict) -> dict:
        """The HSV augmentation transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        hsv_gains = random.uniform(-1, 1, 3) * [self.hue_delta, self.saturation_delta, self.value_delta] + 1
        hue, sat, val = cv2.split(cv2.cvtColor(results['img'], cv2.COLOR_BGR2HSV))

        table_list = np.arange(0, 256, dtype=hsv_gains.dtype)
        lut_hue = ((table_list * hsv_gains[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(table_list * hsv_gains[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(table_list * hsv_gains[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        results['img'] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotations(MMDET_LoadAnnotations):
    """Because the yolo series does not need to consider ignore bboxes for the
    time being, in order to speed up the pipeline, it can be excluded in
    advance."""

    def __init__(self, mask2bbox: bool = False, poly2mask: bool = False, **kwargs) -> None:
        self.mask2bbox = mask2bbox
        assert not poly2mask, 'Does not support BitmapMasks considering ' 'that bitmap consumes more memory.'
        super().__init__(poly2mask=poly2mask, **kwargs)
        if self.mask2bbox:
            assert self.with_mask, 'Using mask2bbox requires ' 'with_mask is True.'
        self._mask_ignore_flag = None

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label and
            semantic segmentation.
        """
        if self.mask2bbox:
            self._load_masks(results)
            if self.with_label:
                self._load_labels(results)
                self._update_mask_ignore_data(results)
            gt_bboxes = results['gt_masks'].get_bboxes(dst_type='hbox')
            results['gt_bboxes'] = gt_bboxes
        else:
            results = super().transform(results)
            self._update_mask_ignore_data(results)
        return results

    def _update_mask_ignore_data(self, results: dict) -> None:
        if 'gt_masks' not in results:
            return

        if 'gt_bboxes_labels' in results and len(results['gt_bboxes_labels']) != len(results['gt_masks']):
            assert len(results['gt_bboxes_labels']) == len(self._mask_ignore_flag)
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][self._mask_ignore_flag]

        if 'gt_bboxes' in results and len(results['gt_bboxes']) != len(results['gt_masks']):
            assert len(results['gt_bboxes']) == len(self._mask_ignore_flag)
            results['gt_bboxes'] = results['gt_bboxes'][self._mask_ignore_flag]

    def _load_bboxes(self, results: dict):
        """Private function to load bounding box annotations.
        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.

        Returns:
            dict: The dict contains loaded bounding box annotations.
        """
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes.append(instance['bbox'])
                gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        if self.box_type is None:
            results['gt_bboxes'] = np.array(gt_bboxes, dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)

    def _load_labels(self, results: dict):
        """Private function to load label annotations.

        Note: BBoxes with ignore_flag of 1 is not considered.
        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        Returns:
            dict: The dict contains loaded label annotations.
        """
        gt_bboxes_labels = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                gt_bboxes_labels.append(instance['bbox_label'])
        results['gt_bboxes_labels'] = np.array(gt_bboxes_labels, dtype=np.int64)

    def _load_masks(self, results: dict) -> None:
        """Private function to load mask annotations.

        Args:
            results (dict): Result dict from :obj:``mmengine.BaseDataset``.
        """
        gt_masks = []
        gt_ignore_flags = []
        self._mask_ignore_flag = []
        for instance in results.get('instances', []):
            if instance['ignore_flag'] == 0:
                if 'mask' in instance:
                    gt_mask = instance['mask']
                    if isinstance(gt_mask, list):
                        gt_mask = [
                            np.array(polygon) for polygon in gt_mask if len(polygon) % 2 == 0 and len(polygon) >= 6
                        ]
                        if len(gt_mask) == 0:
                            # ignore
                            self._mask_ignore_flag.append(0)
                        else:
                            gt_masks.append(gt_mask)
                            gt_ignore_flags.append(instance['ignore_flag'])
                            self._mask_ignore_flag.append(1)
                    else:
                        raise NotImplementedError('Only supports mask annotations in polygon ' 'format currently')
                else:
                    # TODO: Actually, gt with bbox and without mask needs
                    #  to be retained
                    self._mask_ignore_flag.append(0)
        self._mask_ignore_flag = np.array(self._mask_ignore_flag, dtype=bool)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)

        h, w = results['ori_shape']
        gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'mask2bbox={self.mask2bbox}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str


@TRANSFORMS.register_module()
class YOLOv5RandomAffine(BaseTransform):
    """Random affine transform data augmentation in YOLOv5 and YOLOv8. It is
    different from the implementation in YOLOX.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.
    If you set use_mask_refine == True, the code will use the masks
    annotation to refine the bbox.
    Our implementation is slightly different from the official. In COCO
    dataset, a gt may have multiple mask tags.  The official YOLOv5
    annotation file already combines the masks that an object has,
    but our code takes into account the fact that an object has multiple masks.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_masks (PolygonMasks) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)
    - gt_masks (PolygonMasks) (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
        min_bbox_size (float): Width and height threshold to filter bboxes.
            If the height or width of a box is smaller than this value, it
            will be removed. Defaults to 2.
        min_area_ratio (float): Threshold of area ratio between
            original bboxes and wrapped bboxes. If smaller than this value,
            the box will be removed. Defaults to 0.1.
        use_mask_refine (bool): Whether to refine bbox by mask.
        max_aspect_ratio (float): Aspect ratio of width and height
            threshold to filter bboxes. If max(h/w, w/h) larger than this
            value, the box will be removed. Defaults to 20.
        resample_num (int): Number of poly to resample to.
    """

    def __init__(
        self,
        max_rotate_degree: float = 10.0,
        max_translate_ratio: float = 0.1,
        scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
        max_shear_degree: float = 2.0,
        border: Tuple[int, int] = (0, 0),
        border_val: Tuple[int, int, int] = (114, 114, 114),
        bbox_clip_border: bool = True,
        min_bbox_size: int = 2,
        min_area_ratio: float = 0.1,
        use_mask_refine: bool = False,
        max_aspect_ratio: float = 20.0,
        resample_num: int = 1000,
    ):
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border
        self.min_bbox_size = min_bbox_size
        self.min_area_ratio = min_area_ratio
        self.use_mask_refine = use_mask_refine
        self.max_aspect_ratio = max_aspect_ratio
        self.resample_num = resample_num

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """The YOLOv5 random affine transform function.

        Args:
            results (dict): The result dict.

        Returns:
            dict: The result dict.
        """
        img = results['img']
        # self.border is wh format
        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        # Note: Different from YOLOX
        center_matrix = np.eye(3, dtype=np.float32)
        center_matrix[0, 2] = -img.shape[1] / 2
        center_matrix[1, 2] = -img.shape[0] / 2

        warp_matrix, scaling_ratio = self._get_random_homography_matrix(height, width)
        warp_matrix = warp_matrix @ center_matrix

        img = cv2.warpPerspective(img, warp_matrix, dsize=(width, height), borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape
        img_h, img_w = img.shape[:2]

        bboxes = results['gt_bboxes']
        num_bboxes = len(bboxes)
        if num_bboxes:
            orig_bboxes = bboxes.clone()
            if self.use_mask_refine and 'gt_masks' in results:
                # If the dataset has annotations of mask,
                # the mask will be used to refine bbox.
                gt_masks = results['gt_masks']

                gt_masks_resample = self.resample_masks(gt_masks)
                gt_masks = self.warp_mask(gt_masks_resample, warp_matrix, img_h, img_w)

                # refine bboxes by masks
                bboxes = gt_masks.get_bboxes(dst_type='hbox')
                # filter bboxes outside image
                valid_index = self.filter_gt_bboxes(orig_bboxes, bboxes).numpy()
                results['gt_masks'] = gt_masks[valid_index]
            else:
                bboxes.project_(warp_matrix)
                if self.bbox_clip_border:
                    bboxes.clip_([height, width])

                # filter bboxes
                orig_bboxes.rescale_([scaling_ratio, scaling_ratio])

                # Be careful: valid_index must convert to numpy,
                # otherwise it will raise out of bounds when len(valid_index)=1
                valid_index = self.filter_gt_bboxes(orig_bboxes, bboxes).numpy()
                if 'gt_masks' in results:
                    results['gt_masks'] = PolygonMasks(results['gt_masks'].masks, img_h, img_w)

            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_index]

        return results

    @staticmethod
    def warp_poly(poly: np.ndarray, warp_matrix: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
        """Function to warp one mask and filter points outside image.

        Args:
            poly (np.ndarray): Segmentation annotation with shape (n, ) and
                with format (x1, y1, x2, y2, ...).
            warp_matrix (np.ndarray): Affine transformation matrix.
                Shape: (3, 3).
            img_w (int): Width of output image.
            img_h (int): Height of output image.
        """
        # TODO: Current logic may cause retained masks unusable for
        #  semantic segmentation training, which is same as official
        #  implementation.
        poly = poly.reshape((-1, 2))
        poly = np.concatenate((poly, np.ones((len(poly), 1), dtype=poly.dtype)), axis=-1)
        # transform poly
        poly = poly @ warp_matrix.T
        poly = poly[:, :2] / poly[:, 2:3]

        # filter point outside image
        x, y = poly.T
        valid_ind_point = (x >= 0) & (y >= 0) & (x <= img_w) & (y <= img_h)
        return poly[valid_ind_point].reshape(-1)

    def warp_mask(self, gt_masks: PolygonMasks, warp_matrix: np.ndarray, img_w: int, img_h: int) -> PolygonMasks:
        """Warp masks by warp_matrix and retain masks inside image after
        warping.

        Args:
            gt_masks (PolygonMasks): Annotations of semantic segmentation.
            warp_matrix (np.ndarray): Affine transformation matrix.
                Shape: (3, 3).
            img_w (int): Width of output image.
            img_h (int): Height of output image.

        Returns:
            PolygonMasks: Masks after warping.
        """
        masks = gt_masks.masks

        new_masks = []
        for poly_per_obj in masks:
            warpped_poly_per_obj = []
            # One gt may have multiple masks.
            for poly in poly_per_obj:
                valid_poly = self.warp_poly(poly, warp_matrix, img_w, img_h)
                if len(valid_poly):
                    warpped_poly_per_obj.append(valid_poly.reshape(-1))
            # If all the masks are invalid,
            # add [0, 0, 0, 0, 0, 0,] here.
            if not warpped_poly_per_obj:
                # This will be filtered in function `filter_gt_bboxes`.
                warpped_poly_per_obj = [np.zeros(6, dtype=poly_per_obj[0].dtype)]
            new_masks.append(warpped_poly_per_obj)

        gt_masks = PolygonMasks(new_masks, img_h, img_w)
        return gt_masks

    def resample_masks(self, gt_masks: PolygonMasks) -> PolygonMasks:
        """Function to resample each mask annotation with shape (2 * n, ) to
        shape (resample_num * 2, ).

        Args:
            gt_masks (PolygonMasks): Annotations of semantic segmentation.
        """
        masks = gt_masks.masks
        new_masks = []
        for poly_per_obj in masks:
            resample_poly_per_obj = []
            for poly in poly_per_obj:
                poly = poly.reshape((-1, 2))  # xy
                poly = np.concatenate((poly, poly[0:1, :]), axis=0)
                x = np.linspace(0, len(poly) - 1, self.resample_num)
                xp = np.arange(len(poly))
                poly = np.concatenate([np.interp(x, xp, poly[:, i]) for i in range(2)]).reshape(2, -1).T.reshape(-1)
                resample_poly_per_obj.append(poly)
            new_masks.append(resample_poly_per_obj)
        return PolygonMasks(new_masks, gt_masks.height, gt_masks.width)

    def filter_gt_bboxes(self, origin_bboxes: HorizontalBoxes, wrapped_bboxes: HorizontalBoxes) -> torch.Tensor:
        """Filter gt bboxes.

        Args:
            origin_bboxes (HorizontalBoxes): Origin bboxes.
            wrapped_bboxes (HorizontalBoxes): Wrapped bboxes

        Returns:
            dict: The result dict.
        """
        origin_w = origin_bboxes.widths
        origin_h = origin_bboxes.heights
        wrapped_w = wrapped_bboxes.widths
        wrapped_h = wrapped_bboxes.heights
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16), wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h + 1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    @cache_randomness
    def _get_random_homography_matrix(self, height: int, width: int) -> Tuple[np.ndarray, float]:
        """Get random homography matrix.

        Args:
            height (int): Image height.
            width (int): Image width.

        Returns:
            Tuple[np.ndarray, float]: The result of warp_matrix and
            scaling_ratio.
        """
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree, self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0], self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree, self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(0.5 - self.max_translate_ratio, 0.5 + self.max_translate_ratio) * width
        trans_y = random.uniform(0.5 - self.max_translate_ratio, 0.5 + self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)
        warp_matrix = translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix
        return warp_matrix, scaling_ratio

    @staticmethod
    def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
        """Get rotation matrix.

        Args:
            rotate_degrees (float): Rotate degrees.

        Returns:
            np.ndarray: The rotation matrix.
        """
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.0], [np.sin(radian), np.cos(radian), 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        """Get scaling matrix.

        Args:
            scale_ratio (float): Scale ratio.

        Returns:
            np.ndarray: The scaling matrix.
        """
        scaling_matrix = np.array([[scale_ratio, 0.0, 0.0], [0.0, scale_ratio, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float, y_shear_degrees: float) -> np.ndarray:
        """Get shear matrix.

        Args:
            x_shear_degrees (float): X shear degrees.
            y_shear_degrees (float): Y shear degrees.

        Returns:
            np.ndarray: The shear matrix.
        """
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array(
            [[1, np.tan(x_radian), 0.0], [np.tan(y_radian), 1, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )
        return shear_matrix

    @staticmethod
    def _get_translation_matrix(x: float, y: float) -> np.ndarray:
        """Get translation matrix.

        Args:
            x (float): X translation.
            y (float): Y translation.

        Returns:
            np.ndarray: The translation matrix.
        """
        translation_matrix = np.array([[1, 0.0, x], [0.0, 1, y], [0.0, 0.0, 1.0]], dtype=np.float32)
        return translation_matrix

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio_range={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
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
            dataset = results.pop('dataset', None)
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
            assert 'dataset' in results
            # Be careful: deep copying can be very time-consuming
            # if results includes dataset.
            dataset = results.pop('dataset', None)

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
                mix_results = [copy.deepcopy(dataset.get_data_info(index)) for index in indexes]

            if self.pre_transform is not None:
                for i, data in enumerate(mix_results):
                    # pre_transform may also require dataset
                    data.update({'dataset': dataset})
                    # before Mosaic or MixUp need to go through
                    # the necessary pre_transform
                    _results = self.pre_transform(data)
                    _results.pop('dataset')
                    mix_results[i] = _results

            if None not in mix_results:
                results['mix_results'] = mix_results
                break
            print('Repeated calculation')
        else:
            raise RuntimeError(
                'The loading pipeline of the original dataset'
                ' always return None. Please check the correctness '
                'of the dataset and its pipeline.'
            )

        # Mosaic or MixUp
        results = self.mix_img_transform(results)

        if 'mix_results' in results:
            results.pop('mix_results')
        results['dataset'] = dataset

        return results


@TRANSFORMS.register_module()
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
        use_cached: bool = False,
        max_cached_images: int = 40,
        random_pop: bool = True,
        max_refetch: int = 15,
    ):
        assert isinstance(img_scale, tuple)
        assert 0 <= prob <= 1.0, 'The probability should be in range [0,1]. ' f'got {prob}.'
        if use_cached:
            assert max_cached_images >= 4, 'The length of cache must >= 4, ' f'but got {max_cached_images}.'

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
        assert 'mix_results' in results
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        mosaic_masks = []
        with_mask = True if 'gt_masks' in results else False
        # self.img_scale is wh format
        img_scale_w, img_scale_h = self.img_scale

        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(img_scale_h * 2), int(img_scale_w * 2), 3), self.pad_val, dtype=results['img'].dtype
            )
        else:
            mosaic_img = np.full((int(img_scale_h * 2), int(img_scale_w * 2)), self.pad_val, dtype=results['img'].dtype)

        # mosaic center x, y
        center_x = int(random.uniform(*self.center_ratio_range) * img_scale_w)
        center_y = int(random.uniform(*self.center_ratio_range) * img_scale_h)
        center_position = (center_x, center_y)

        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                results_patch = results
            else:
                results_patch = results['mix_results'][i - 1]

            img_i = results_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(img_scale_h / h_i, img_scale_w / w_i)
            img_i = mmcv.imresize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord

            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = results_patch['gt_bboxes']
            gt_bboxes_labels_i = results_patch['gt_bboxes_labels']
            gt_ignore_flags_i = results_patch['gt_ignore_flags']

            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
            if with_mask and results_patch.get('gt_masks', None) is not None:
                gt_masks_i = results_patch['gt_masks']
                gt_masks_i = gt_masks_i.rescale(float(scale_ratio_i))
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                    offset=padw,
                    direction='horizontal',
                )
                gt_masks_i = gt_masks_i.translate(
                    out_shape=(int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                    offset=padh,
                    direction='vertical',
                )
                mosaic_masks.append(gt_masks_i)

        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)

        if self.bbox_clip_border:
            mosaic_bboxes.clip_([2 * img_scale_h, 2 * img_scale_w])
            if with_mask:
                mosaic_masks = mosaic_masks[0].cat(mosaic_masks)
                results['gt_masks'] = mosaic_masks
        else:
            # remove outside bboxes
            inside_inds = mosaic_bboxes.is_inside([2 * img_scale_h, 2 * img_scale_w]).numpy()
            mosaic_bboxes = mosaic_bboxes[inside_inds]
            mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
            mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]
            if with_mask:
                mosaic_masks = mosaic_masks[0].cat(mosaic_masks)[inside_inds]
                results['gt_masks'] = mosaic_masks

        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags

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
        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                max(center_position_xy[1] - img_shape_wh[1], 0),
                center_position_xy[0],
                center_position_xy[1],
            )
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                max(center_position_xy[1] - img_shape_wh[1], 0),
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[0] * 2),
                center_position_xy[1],
            )
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = (
                max(center_position_xy[0] - img_shape_wh[0], 0),
                center_position_xy[1],
                center_position_xy[0],
                min(self.img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = (
                center_position_xy[0],
                center_position_xy[1],
                min(center_position_xy[0] + img_shape_wh[0], self.img_scale[0] * 2),
                min(self.img_scale[1] * 2, center_position_xy[1] + img_shape_wh[1]),
            )
            crop_coord = 0, 0, min(img_shape_wh[0], x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob})'
        return repr_str
