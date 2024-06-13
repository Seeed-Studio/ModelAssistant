from typing import Tuple
import math

import torch
import numpy as np
from numpy import random

import cv2
from mmcv.transforms import BaseTransform


from mmcv.transforms.utils import cache_randomness
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import PolygonMasks

from sscma.registry import TRANSFORMS


@TRANSFORMS.register_module()
class YOLOv5RandomAffine(BaseTransform):
    """Random affine transform data augmentation in YOLOv5 and YOLOv8. It is
    different from the implementation in YOLOX.

    Copyright (c) OpenMMLab.

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
        use_mask_refine (bool): Whether to refine bbox by mask. Deprecated.
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
        # The use_mask_refine parameter has been deprecated.
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
            orig_bboxes.rescale_([scaling_ratio, scaling_ratio])
            if 'gt_masks' in results:
                # If the dataset has annotations of mask,
                # the mask will be used to refine bbox.
                gt_masks = results['gt_masks']

                gt_masks_resample = self.resample_masks(gt_masks)
                gt_masks = self.warp_mask(gt_masks_resample, warp_matrix, img_h, img_w)

                # refine bboxes by masks
                bboxes = self.segment2box(gt_masks, height, width)
                # filter bboxes outside image
                valid_index = self.filter_gt_bboxes(orig_bboxes, bboxes).numpy()
                if self.bbox_clip_border:
                    bboxes.clip_([height - 1e-3, width - 1e-3])
                    gt_masks = self.clip_polygons(gt_masks, height, width)
                results['gt_masks'] = gt_masks[valid_index]
            else:
                bboxes.project_(warp_matrix)
                if self.bbox_clip_border:
                    bboxes.clip_([height, width])

                # filter bboxes
                # Be careful: valid_index must convert to numpy,
                # otherwise it will raise out of bounds when len(valid_index)=1
                valid_index = self.filter_gt_bboxes(orig_bboxes, bboxes).numpy()

            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_index]
        else:
            if 'gt_masks' in results:
                results['gt_masks'] = PolygonMasks([], img_h, img_w)

        return results

    def segment2box(self, gt_masks: PolygonMasks, height: int, width: int) -> HorizontalBoxes:
        """
        Convert 1 segment label to 1 box label, applying inside-image
        constraint i.e. (xy1, xy2, ...) to (xyxy)
        Args:
            gt_masks (torch.Tensor): the segment label
            width (int): the width of the image. Defaults to 640
            height (int): The height of the image. Defaults to 640
        Returns:
            HorizontalBoxes: the clip bboxes from gt_masks.
        """
        bboxes = []
        for _, poly_per_obj in enumerate(gt_masks):
            # simply use a number that is big enough for comparison with
            # coordinates
            xy_min = np.array([width * 2, height * 2], dtype=np.float32)
            xy_max = np.zeros(2, dtype=np.float32) - 1

            for p in poly_per_obj:
                xy = np.array(p).reshape(-1, 2).astype(np.float32)
                x, y = xy.T
                inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
                x, y = x[inside], y[inside]
                if not any(x):
                    continue
                xy = np.stack([x, y], axis=0).T

                xy_min = np.minimum(xy_min, np.min(xy, axis=0))
                xy_max = np.maximum(xy_max, np.max(xy, axis=0))
            if xy_max[0] == -1:
                bbox = np.zeros(4, dtype=np.float32)
            else:
                bbox = np.concatenate([xy_min, xy_max], axis=0)
            bboxes.append(bbox)

        return HorizontalBoxes(np.stack(bboxes, axis=0))

    # TODO: Move to mmdet
    def clip_polygons(self, gt_masks: PolygonMasks, height: int, width: int) -> PolygonMasks:
        """Function to clip points of polygons with height and width.

        Args:
            gt_masks (PolygonMasks): Annotations of instance segmentation.
            height (int): height of clip border.
            width (int): width of clip border.
        Return:
            clipped_masks (PolygonMasks):
                Clip annotations of instance segmentation.
        """
        if len(gt_masks) == 0:
            clipped_masks = PolygonMasks([], height, width)
        else:
            clipped_masks = []
            for poly_per_obj in gt_masks:
                clipped_poly_per_obj = []
                for p in poly_per_obj:
                    p = p.copy()
                    p[0::2] = p[0::2].clip(0, width)
                    p[1::2] = p[1::2].clip(0, height)
                    clipped_poly_per_obj.append(p)
                clipped_masks.append(clipped_poly_per_obj)
            clipped_masks = PolygonMasks(clipped_masks, height, width)
        return clipped_masks

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

        return poly.reshape(-1)

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
