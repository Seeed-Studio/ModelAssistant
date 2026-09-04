# Copyright (c) Seeed Technology Co.,Ltd.
# Copyright (c) OpenMMLab.
"""YOLOv5-style data transforms, ported from the 2.0.0 branch.

Registered by direct class reference in configs (main-branch style); the
mmcv image ops are replaced by the vendored simplecv equivalents.
"""

import copy
import math
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from numpy import random

from mmengine.utils import is_str

from sscma.structures.bbox import HorizontalBoxes, autocast_box_type
from sscma.structures.mask import BitmapMasks, PolygonMasks
from sscma.utils.simplecv import simplecv_impad, simplecv_imrescale, simplecv_imresize

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from .transforms import Resize
from .utils import BaseTransform, avoid_cache_randomness, cache_randomness

class YOLOv5KeepRatioResize(Resize):
    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 keep_ratio: bool=True,
                 **kwargs):
        assert keep_ratio is True
        super().__init__(scale=scale, keep_ratio=True, **kwargs)
        
    @staticmethod
    def _get_ratio(old_size: Tuple[int, int],
                   scale: Union[float, Tuple[int, int]]) -> float:
        w, h = old_size
        if isinstance(scale, (float, int)):
            ratio = scale
        elif isinstance(scale, Tuple):
            max_long_edge = max(scale)
            max_short_edge = min(scale)
            ratio = min(max_long_edge / max(h, w), max_short_edge / min(h, w))
        return ratio

    def _resize_img(self, result: dict):
        assert self.keep_ratio is True
        if result.get('img', None) is not None:
            img = result['img']
            original_h, original_w = img.shape[:2]
            ratio = self._get_ratio((original_h, original_w), self.scale)
            if ratio != 1:
                img = simplecv_imrescale(img=img,
                                     scale=ratio,
                                     interpolation='area' if ratio < 1 else 'bilinear',
                                     backend=self.backend)
            resized_h, resized_W = img.shape[:2]
            scale_ratio = resized_h / original_h
            scale_factor = (scale_ratio, scale_ratio)

            result['img'] = img
            result['img_shape'] = img.shape[:2]
            result['scale_factor'] = scale_factor



class LetterResize(Resize):
    def __init__(self,
                 scale: Union[int, Tuple[int, int]],
                 pad_val: dict = dict(img=0, mask=0, seg=255),
                 use_mini_pad: bool = False,
                 stretch_only: bool = False,
                 allow_scale_up: bool = True,
                 **kwargs):
        super().__init__(scale=scale, keep_ratio=True, **kwargs)

        self.pad_val = pad_val
        if isinstance(pad_val, (int, float)):
            pad_val = dict(img=pad_val, seg=255)

        self.use_mini_pad = use_mini_pad
        self.stretch_only = stretch_only
        self.allow_scale_up = allow_scale_up

    def _resize_img(self, result: dict):
        img = result.get('img', None)
        if img is None:
            warnings.warn("Get no image in resize process", UserWarning)
            return

        if 'batch_shape' in result:
            scale = tuple(result['batch_shape'])
        else:
            scale = self.scale[::-1]

        img_shape = img.shape[:2]
        ratio = min(scale[0] / img_shape[0], scale[1] / img_shape[1])

        if not self.allow_scale_up:
            ratio = min(ratio, 1.0)

        ratio = [ratio, ratio]

        prev_pad_shape = (int(round(img.shape[0] * ratio[0])),
                          int(round(img_shape[1] * ratio[1])))
        
        pad_h, pad_w = [scale[0] - prev_pad_shape[0], scale[1]-prev_pad_shape[1]]
        
        if self.use_mini_pad:
            pad_h, pad_w = np.mod(pad_h, 32), np.mod(pad_w, 32)
        elif self.stretch_only:
            pad_h, pad_w = 0.0, 0.0
            prev_pad_shape = (scale[0], scale[1])
            ratio = [scale[0]/img.shape[0], scale[1]/img.shape[1]]

        # resize receives size (w, h)
        if img.shape != prev_pad_shape:
            img = simplecv_imresize(img, (prev_pad_shape[1], prev_pad_shape[0]),
                                interpolation=self.interpolation, backend=self.backend)
        
        scale_factor = (ratio[1], ratio[0])  # (w, h)
        if 'scale_factor' in result: 
            result['scale_factor_origin'] = result['scale_factor']
        result['scale_factor'] = scale_factor

        top_padding, left_padding = int(round(pad_h // 2 - 0.1)), int(round(pad_w // 2 - 0.1))
        bottom_padding = pad_h - top_padding
        right_padding = pad_w - left_padding

        padding_list = [top_padding, bottom_padding, left_padding, right_padding]

        if top_padding != 0 or bottom_padding != 0 or \
                left_padding != 0 or right_padding != 0:

            pad_val = self.pad_val.get('img', 0)
            if isinstance(pad_val, int) and img.ndim == 3:
                pad_val = tuple(pad_val for _ in range(img.shape[2]))

            img = simplecv_impad(
                img=img,
                padding=(padding_list[2], padding_list[0], padding_list[3],
                         padding_list[1]),
                pad_val=pad_val,
                padding_mode='constant')
        result['img'] = img
        result['img_shape'] = img.shape
        if 'pad_param' in result:
            result['pad_param_origin'] = result['pad_param'] * \
                                          np.repeat(ratio, 2)
        result['pad_param'] = np.array(padding_list, dtype=np.float32)

    def _resize_masks(self, result: dict):
        if result.get('gt_masks', None) is None:
            return
        
        gt_masks = result['gt_masks']
        assert isinstance(gt_masks, PolygonMasks), f'Only supports PolygonMasks, but got {type(gt_masks)}'
        
        gt_mask_h = result['gt_masks'].height * result['scale_factor'][1]
        gt_mask_w = result['gt_masks'].weight * result['scale_factor'][0]
        gt_masks = result['gt_masks'].resize((int(round(gt_mask_h)), int(round(gt_mask_w))))

        top_padding, _, left_padding, _ = result['pad_param']
        if int(left_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=result['img_shape'][:2],
                offset=int(left_padding),
                direction='horizontal')
        if int(top_padding) != 0:
            gt_masks = gt_masks.translate(
                out_shape=result['img_shape'][:2],
                offset=int(top_padding),
                direction='vertical')
        result['gt_masks'] = gt_masks
    
    def _resize_bboxes(self, result: dict) -> None:
        if result.get('gt_bboxes', None) is None:
            return
        result['gt_bboxes'].rescale_(result['scale_factor'])

        if len(result['pad_param']) != 4:
            return
        result['gt_bboxes'].translate_((result['pad_param'][2], result['pad_param'][0]))
        if self.clip_object_border:
            result['gt_bboxes'].clip_(result['img_shape'])
    
    def transform(self, result: dict) -> dict:
        result = super().transform(result)
        if 'scale_factor_origin' in result:
            scale_factor_origin = result.pop('scale_factor_origin')
            result['scale_factor'] = (result['scale_factor'][0] *
                                       scale_factor_origin[0],
                                       result['scale_factor'][1] *
                                       scale_factor_origin[1])
        if 'pad_param_origin' in result:
            pad_param_origin = result.pop('pad_param_origin')
            result['pad_param'] += pad_param_origin
        return result
    


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

    def __init__(self,
                 hue_delta: Union[int, float] = 0.015,
                 saturation_delta: Union[int, float] = 0.7,
                 value_delta: Union[int, float] = 0.4):
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
        hsv_gains = \
            random.uniform(-1, 1, 3) * \
            [self.hue_delta, self.saturation_delta, self.value_delta] + 1
        hue, sat, val = cv2.split(
            cv2.cvtColor(results['img'], cv2.COLOR_BGR2HSV))

        table_list = np.arange(0, 256, dtype=hsv_gains.dtype)
        lut_hue = ((table_list * hsv_gains[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(table_list * hsv_gains[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(table_list * hsv_gains[2], 0, 255).astype(np.uint8)

        im_hsv = cv2.merge(
            (cv2.LUT(hue, lut_hue), cv2.LUT(sat,
                                            lut_sat), cv2.LUT(val, lut_val)))
        results['img'] = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str



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

    def __init__(self,
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
                 max_aspect_ratio: float = 20.,
                 resample_num: int = 1000):
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

        warp_matrix, scaling_ratio = self._get_random_homography_matrix(
            height, width)
        warp_matrix = warp_matrix @ center_matrix

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
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
                gt_masks = self.warp_mask(gt_masks_resample, warp_matrix,
                                          img_h, img_w)

                # refine bboxes by masks
                bboxes = gt_masks.get_bboxes(dst_type='hbox')
                # filter bboxes outside image
                valid_index = self.filter_gt_bboxes(orig_bboxes,
                                                    bboxes).numpy()
                results['gt_masks'] = gt_masks[valid_index]
            else:
                bboxes.project_(warp_matrix)
                if self.bbox_clip_border:
                    bboxes.clip_([height, width])

                # filter bboxes
                orig_bboxes.rescale_([scaling_ratio, scaling_ratio])

                # Be careful: valid_index must convert to numpy,
                # otherwise it will raise out of bounds when len(valid_index)=1
                valid_index = self.filter_gt_bboxes(orig_bboxes,
                                                    bboxes).numpy()
                if 'gt_masks' in results:
                    results['gt_masks'] = PolygonMasks(
                        results['gt_masks'].masks, img_h, img_w)

            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]

        return results

    @staticmethod
    def warp_poly(poly: np.ndarray, warp_matrix: np.ndarray, img_w: int,
                  img_h: int) -> np.ndarray:
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
        poly = np.concatenate((poly, np.ones(
            (len(poly), 1), dtype=poly.dtype)),
                              axis=-1)
        # transform poly
        poly = poly @ warp_matrix.T
        poly = poly[:, :2] / poly[:, 2:3]

        # filter point outside image
        x, y = poly.T
        valid_ind_point = (x >= 0) & (y >= 0) & (x <= img_w) & (y <= img_h)
        return poly[valid_ind_point].reshape(-1)

    def warp_mask(self, gt_masks: PolygonMasks, warp_matrix: np.ndarray,
                  img_w: int, img_h: int) -> PolygonMasks:
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
                warpped_poly_per_obj = [
                    np.zeros(6, dtype=poly_per_obj[0].dtype)
                ]
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
                poly = np.concatenate([
                    np.interp(x, xp, poly[:, i]) for i in range(2)
                ]).reshape(2, -1).T.reshape(-1)
                resample_poly_per_obj.append(poly)
            new_masks.append(resample_poly_per_obj)
        return PolygonMasks(new_masks, gt_masks.height, gt_masks.width)

    def filter_gt_bboxes(self, origin_bboxes: HorizontalBoxes,
                         wrapped_bboxes: HorizontalBoxes) -> torch.Tensor:
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
        aspect_ratio = np.maximum(wrapped_w / (wrapped_h + 1e-16),
                                  wrapped_h / (wrapped_w + 1e-16))

        wh_valid_idx = (wrapped_w > self.min_bbox_size) & \
                       (wrapped_h > self.min_bbox_size)
        area_valid_idx = wrapped_w * wrapped_h / (origin_w * origin_h +
                                                  1e-16) > self.min_area_ratio
        aspect_ratio_valid_idx = aspect_ratio < self.max_aspect_ratio
        return wh_valid_idx & area_valid_idx & aspect_ratio_valid_idx

    @cache_randomness
    def _get_random_homography_matrix(self, height: int,
                                      width: int) -> Tuple[np.ndarray, float]:
        """Get random homography matrix.

        Args:
            height (int): Image height.
            width (int): Image width.

        Returns:
            Tuple[np.ndarray, float]: The result of warp_matrix and
            scaling_ratio.
        """
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * width
        trans_y = random.uniform(0.5 - self.max_translate_ratio,
                                 0.5 + self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)
        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
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
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
        """Get scaling matrix.

        Args:
            scale_ratio (float): Scale ratio.

        Returns:
            np.ndarray: The scaling matrix.
        """
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix

    @staticmethod
    def _get_shear_matrix(x_shear_degrees: float,
                          y_shear_degrees: float) -> np.ndarray:
        """Get shear matrix.

        Args:
            x_shear_degrees (float): X shear degrees.
            y_shear_degrees (float): Y shear degrees.

        Returns:
            np.ndarray: The shear matrix.
        """
        x_radian = math.radians(x_shear_degrees)
        y_radian = math.radians(y_shear_degrees)
        shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                 [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                                dtype=np.float32)
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
        translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                      dtype=np.float32)
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


@avoid_cache_randomness
class Albu(BaseTransform):
    """Albumentation augmentation.

    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.

    Required Keys:

    - img (np.uint8)
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)

    Modified Keys:

    - img (np.uint8)
    - gt_bboxes (HorizontalBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - img_shape (tuple)

    An example of ``transforms`` is as followed:

    .. code-block::

        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]

    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict, optional): Bbox_params for albumentation `Compose`
        keymap (dict, optional): Contains
            {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug. Defaults to False.
    """

    def __init__(self,
                 transforms: List[dict],
                 bbox_params: Optional[dict] = None,
                 keymap: Optional[dict] = None,
                 skip_img_without_anno: bool = False) -> None:
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        # Args will be modified later, copying it will be safer
        transforms = copy.deepcopy(transforms)
        if bbox_params is not None:
            bbox_params = copy.deepcopy(bbox_params)
        if keymap is not None:
            keymap = copy.deepcopy(keymap)
        self.transforms = transforms
        self.filter_lost_elements = False
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg: dict) -> albumentations:
        """Import a module from albumentations.

        It inherits some of :func:`build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed object.
        """

        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()
        obj_type = args.pop('type')
        if is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d: dict, keymap: dict) -> dict:
        """Dictionary mapper. Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function of Albu."""
        # TODO: gt_seg_map is not currently supported
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)
        results, ori_masks = self._preprocess_results(results)
        results = self.aug(**results)
        results = self._postprocess_results(results, ori_masks)
        if results is None:
            return None
        # back to the original format
        results = self.mapper(results, self.keymap_back)
        results['img_shape'] = results['img'].shape[:2]
        return results

    def _preprocess_results(self, results: dict) -> tuple:
        """Pre-processing results to facilitate the use of Albu."""
        if 'bboxes' in results:
            # to list of boxes
            if not isinstance(results['bboxes'], HorizontalBoxes):
                raise NotImplementedError(
                    'Albu only supports horizontal boxes now')
            bboxes = results['bboxes'].numpy()
            results['bboxes'] = [x for x in bboxes]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        # TODO: Support mask structure in albu
        ori_masks = None
        if 'masks' in results:
            if isinstance(results['masks'], PolygonMasks):
                raise NotImplementedError(
                    'Albu only supports BitMap masks now')
            ori_masks = results['masks']
            if albumentations.__version__ < '0.5':
                results['masks'] = results['masks'].masks
            else:
                results['masks'] = [mask for mask in results['masks'].masks]

        return results, ori_masks

    def _postprocess_results(
            self,
            results: dict,
            ori_masks: Optional[Union[BitmapMasks,
                                      PolygonMasks]] = None) -> dict:
        """Post-processing Albu output."""
        # albumentations may return np.array or list on different versions
        if 'gt_bboxes_labels' in results and isinstance(
                results['gt_bboxes_labels'], list):
            results['gt_bboxes_labels'] = np.array(
                results['gt_bboxes_labels'], dtype=np.int64)
        if 'gt_ignore_flags' in results and isinstance(
                results['gt_ignore_flags'], list):
            results['gt_ignore_flags'] = np.array(
                results['gt_ignore_flags'], dtype=bool)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)
            results['bboxes'] = HorizontalBoxes(results['bboxes'])

            # filter label_fields
            if self.filter_lost_elements:

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    assert ori_masks is not None
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])
                    results['masks'] = ori_masks.__class__(
                        results['masks'], ori_masks.height, ori_masks.width)

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None
            elif 'masks' in results:
                results['masks'] = ori_masks.__class__(results['masks'],
                                                       ori_masks.height,
                                                       ori_masks.width)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str
