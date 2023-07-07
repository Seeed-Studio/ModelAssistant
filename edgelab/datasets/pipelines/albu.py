from typing import Optional, Sequence, Tuple, Union

import cv2
import albumentations as A
from edgelab.registry import TRANSFORMS


@TRANSFORMS.register_module()
class ColorJitter(A.ColorJitter):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5):
        super().__init__(brightness, contrast, saturation, hue, always_apply, p)


@TRANSFORMS.register_module()
class HorizontalFlip(A.HorizontalFlip):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)


@TRANSFORMS.register_module()
class VerticalFlip(A.VerticalFlip):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)


@TRANSFORMS.register_module()
class Rotate(A.Rotate):
    def __init__(
        self,
        limit=90,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_REFLECT_101,
        value=None,
        mask_value=None,
        rotate_method="largest_box",
        crop_border=False,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(
            limit, interpolation, border_mode, value, mask_value, rotate_method, crop_border, always_apply, p
        )


@TRANSFORMS.register_module()
class Affine(A.Affine):
    def __init__(
        self,
        scale: Optional[Union[float, Sequence[float], dict]] = None,
        translate_percent: Optional[Union[float, Sequence[float], dict]] = None,
        translate_px: Optional[Union[int, Sequence[int], dict]] = None,
        rotate: Optional[Union[float, Sequence[float]]] = None,
        shear: Optional[Union[float, Sequence[float], dict]] = None,
        interpolation: int = cv2.INTER_LINEAR,
        mask_interpolation: int = cv2.INTER_NEAREST,
        cval: Union[int, float, Sequence[int], Sequence[float]] = 0,
        cval_mask: Union[int, float, Sequence[int], Sequence[float]] = 0,
        mode: int = cv2.BORDER_CONSTANT,
        fit_output: bool = False,
        keep_ratio: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            scale,
            translate_percent,
            translate_px,
            rotate,
            shear,
            interpolation,
            mask_interpolation,
            cval,
            cval_mask,
            mode,
            fit_output,
            keep_ratio,
            always_apply,
            p,
        )


@TRANSFORMS.register_module()
class ChannelShuffle(A.ChannelShuffle):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)


@TRANSFORMS.register_module()
class OneOf(A.OneOf):
    def __init__(self, transforms, p: float = 0.5):
        super().__init__(transforms, p)


@TRANSFORMS.register_module()
class IAAAdditiveGaussianNoise(A.IAAAdditiveGaussianNoise):
    def __init__(self, loc=0, scale=..., per_channel=False, always_apply=False, p=0.5):
        super().__init__(loc, scale, per_channel, always_apply, p)


@TRANSFORMS.register_module()
class GaussNoise(A.GaussNoise):
    def __init__(self, var_limit=..., mean=0, per_channel=True, always_apply=False, p=0.5):
        super().__init__(var_limit, mean, per_channel, always_apply, p)


@TRANSFORMS.register_module()
class Blur(A.Blur):
    def __init__(self, blur_limit=7, always_apply: bool = False, p: float = 0.5):
        super().__init__(blur_limit, always_apply, p)


@TRANSFORMS.register_module()
class MotionBlur(A.MotionBlur):
    def __init__(self, blur_limit=7, allow_shifted: bool = True, always_apply: bool = False, p: float = 0.5):
        super().__init__(blur_limit, allow_shifted, always_apply, p)


@TRANSFORMS.register_module()
class MedianBlur(A.MedianBlur):
    def __init__(self, blur_limit, always_apply: bool = False, p: float = 0.5):
        super().__init__(blur_limit, always_apply, p)


@TRANSFORMS.register_module()
class SafeRotate(A.SafeRotate):
    def __init__(
        self,
        limit: Union[float, Tuple[float, float]] = 90,
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value=None,
        mask_value: Optional[Union[int, float, Sequence[int], Sequence[float]]] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(limit, interpolation, border_mode, value, mask_value, always_apply, p)


@TRANSFORMS.register_module()
class RandomCrop(A.RandomCrop):
    def __init__(self, height, width, always_apply=False, p=1):
        super().__init__(height, width, always_apply, p)


@TRANSFORMS.register_module()
class Resize(A.Resize):
    def __init__(self, height, width, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1):
        super().__init__(height, width, interpolation, always_apply, p)


@TRANSFORMS.register_module()
class ToGray(A.ToGray):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)


@TRANSFORMS.register_module()
class CoarseDropout(A.CoarseDropout):
    def __init__(
        self,
        max_holes: int = 8,
        max_height: int = 8,
        max_width: int = 8,
        min_holes: Optional[int] = None,
        min_height: Optional[int] = None,
        min_width: Optional[int] = None,
        fill_value: int = 0,
        mask_fill_value: Optional[int] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            max_holes,
            max_height,
            max_width,
            min_holes,
            min_height,
            min_width,
            fill_value,
            mask_fill_value,
            always_apply,
            p,
        )


@TRANSFORMS.register_module()
class CoraseDropout(A.CoarseDropout):
    def __init__(
        self,
        max_holes: int = 8,
        max_height: int = 8,
        max_width: int = 8,
        min_holes: Optional[int] = None,
        min_height: Optional[int] = None,
        min_width: Optional[int] = None,
        fill_value: int = 0,
        mask_fill_value: Optional[int] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            max_holes,
            max_height,
            max_width,
            min_holes,
            min_height,
            min_width,
            fill_value,
            mask_fill_value,
            always_apply,
            p,
        )


@TRANSFORMS.register_module()
class RandomResizedCrop(A.RandomResizedCrop):
    def __init__(
        self,
        height,
        width,
        scale=(0.08, 1.0),
        ratio=(0.75, 1.3333333333333333),
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=1.0,
    ):
        super().__init__(height, width, scale, ratio, interpolation, always_apply, p)


@TRANSFORMS.register_module()
class RandomBrightnessContrast(A.RandomBrightnessContrast):
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5):
        super().__init__(brightness_limit, contrast_limit, brightness_by_max, always_apply, p)
