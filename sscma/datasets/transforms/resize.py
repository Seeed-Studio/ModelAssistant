from typing import Tuple, Union

import numpy as np

import mmcv

from mmdet.datasets.transforms import Resize as MMDET_Resize
from mmdet.structures.mask import PolygonMasks

from sscma.registry import TRANSFORMS


@TRANSFORMS.register_module()
class YOLOv5KeepRatioResize(MMDET_Resize):
    """Resize images & bbox(if existed).

    Copyright (c) OpenMMLab.

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
                # resize image according to the shape
                # NOTE: We are currently testing on COCO that modifying
                # this code will not affect the results.
                # If you find that it has an effect on your results,
                # please feel free to contact us.
                image = mmcv.imresize(
                    img=image,
                    size=(int(original_w * ratio), int(original_h * ratio)),
                    interpolation='area' if ratio < 1 else 'bilinear',
                    backend=self.backend,
                )

            resized_h, resized_w = image.shape[:2]
            scale_ratio_h = resized_h / original_h
            scale_ratio_w = resized_w / original_w
            scale_factor = (scale_ratio_w, scale_ratio_h)

            results['img'] = image
            results['img_shape'] = image.shape[:2]
            results['scale_factor'] = scale_factor


@TRANSFORMS.register_module()
class LetterResize(MMDET_Resize):
    """Resize and pad image while meeting stride-multiple constraints.

    Copyright (c) OpenMMLab.

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
        half_pad_param (bool): If set to True, left and right pad_param will
            be given by dividing padding_h by 2. If set to False, pad_param is
            in int format. We recommend setting this to False for object
            detection tasks, and True for instance segmentation tasks.
            Default to False.
    """

    def __init__(
        self,
        scale: Union[int, Tuple[int, int]],
        pad_val: dict = dict(img=0, mask=0, seg=255),
        use_mini_pad: bool = False,
        stretch_only: bool = False,
        allow_scale_up: bool = True,
        half_pad_param: bool = False,
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
        self.half_pad_param = half_pad_param

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

        scale_factor = (no_pad_shape[1] / image_shape[1], no_pad_shape[0] / image_shape[0])

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

        if self.half_pad_param:
            results['pad_param'] = np.array(
                [padding_h / 2, padding_h / 2, padding_w / 2, padding_w / 2], dtype=np.float32
            )
        else:
            # We found in object detection, using padding list with
            # int type can get higher mAP.
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
