from typing import Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
from mmcv.transforms.utils import cache_randomness

from sscma.registry import TRANSFORMS

from .base import BaseTransform


class BaseAugTransform(BaseTransform):
    r"""The base class of augmentation transform for RandAugment.

    This class provides several common attributions and methods to support the
    magnitude level mapping and magnitude level randomness in
    :class:`RandAugment`.

    Args:
        magnitude_level (int | float): Magnitude level.
        magnitude_range (Sequence[number], optional): For augmentation have
            magnitude argument, maybe "magnitude", "angle" or other, you can
            specify the magnitude level mapping range to generate the magnitude
            argument. For example, assume ``total_level`` is 10,
            ``magnitude_level=3`` specify magnitude is 3 if
            ``magnitude_range=(0, 10)`` while specify magnitude is 7 if
            ``magnitude_range=(10, 0)``. Defaults to None.
        magnitude_std (Number | str): Deviation of magnitude noise applied.

            - If positive number, the magnitude obeys normal distribution
              :math:`\mathcal{N}(magnitude, magnitude_std)`.
            - If 0 or negative number, magnitude remains unchanged.
            - If str "inf", the magnitude obeys uniform distribution
              :math:`Uniform(min, magnitude)`.

            Defaults to 0.
        total_level (int | float): Total level for the magnitude. Defaults to
            10.
        prob (float): The probability for performing transformation therefore
            should be in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the magnitude
            negative, which should be in range [0,1]. Defaults to 0.
    """

    def __init__(
        self,
        magnitude_level: int = 10,
        magnitude_range: Tuple[float, float] = None,
        magnitude_std: Union[str, float] = 0.0,
        total_level: int = 10,
        prob: float = 0.5,
        random_negative_prob: float = 0.5,
    ):
        self.magnitude_level = magnitude_level
        self.magnitude_range = magnitude_range
        self.magnitude_std = magnitude_std
        self.total_level = total_level
        self.prob = prob
        self.random_negative_prob = random_negative_prob

    @cache_randomness
    def random_disable(self):
        """Randomly disable the transform."""
        return np.random.rand() > self.prob

    @cache_randomness
    def random_magnitude(self):
        """Randomly generate magnitude."""
        magnitude = self.magnitude_level
        # if magnitude_std is positive number or 'inf', move
        # magnitude_value randomly.
        if self.magnitude_std == 'inf':
            magnitude = np.random.uniform(0, magnitude)
        elif self.magnitude_std > 0:
            magnitude = np.random.normal(magnitude, self.magnitude_std)
            magnitude = np.clip(magnitude, 0, self.total_level)

        val1, val2 = self.magnitude_range
        magnitude = (magnitude / self.total_level) * (val2 - val1) + val1
        return magnitude

    @cache_randomness
    def random_negative(self, value):
        """Randomly negative the value."""
        if np.random.rand() < self.random_negative_prob:
            return -value
        else:
            return value

    def extra_repr(self):
        """Extra repr string when auto-generating magnitude is enabled."""
        if self.magnitude_range is not None:
            repr_str = f', magnitude_level={self.magnitude_level}, '
            repr_str += f'magnitude_range={self.magnitude_range}, '
            repr_str += f'magnitude_std={self.magnitude_std}, '
            repr_str += f'total_level={self.total_level}, '
            return repr_str
        else:
            return ''


@TRANSFORMS.register_module()
class RandomRotate(BaseAugTransform):
    """Rotate images.

    Args:
        angle (float, optional): The angle used for rotate. Positive values
            stand for clockwise rotation. If None, generate from
            ``magnitude_range``, see :class:`BaseAugTransform`.
            Defaults to None.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If None, the center of the image will be used.
            Defaults to None.
        scale (float): Isotropic scale factor. Defaults to 1.0.
        pad_val (int, Sequence[int]): Pixel pad_val value for constant fill.
            If a sequence of length 3, it is used to pad_val R, G, B channels
            respectively. Defaults to 128.
        prob (float): The probability for performing rotate therefore should be
            in range [0, 1]. Defaults to 0.5.
        random_negative_prob (float): The probability that turns the angle
            negative, which should be in range [0,1]. Defaults to 0.5.
        interpolation (str): Interpolation method. Options are 'nearest',
            'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to 'nearest'.
        **kwargs: Other keyword arguments of :class:`BaseAugTransform`.
    """

    def __init__(
        self,
        angle: Optional[float] = None,
        center: Optional[Tuple[float]] = None,
        scale: float = 1.0,
        pad_val: Union[int, Sequence[int]] = 128,
        prob: float = 0.5,
        random_negative_prob: float = 0.5,
        interpolation: str = 'nearest',
        **kwargs,
    ):
        super().__init__(prob=prob, random_negative_prob=random_negative_prob, **kwargs)
        assert (angle is None) ^ (
            self.magnitude_range is None
        ), 'Please specify only one of `angle` and `magnitude_range`.'

        self.angle = angle
        self.center = center
        self.scale = scale
        if isinstance(pad_val, Sequence):
            self.pad_val = tuple(pad_val)
        else:
            self.pad_val = pad_val

        self.interpolation = interpolation

    def transform(self, results):
        """Apply transform to results."""
        if self.random_disable():
            return results

        if self.angle is not None:
            angle = self.random_negative(self.angle)
        else:
            angle = self.random_negative(self.random_magnitude())

        img = results['img']
        img_rotated = mmcv.imrotate(
            img,
            angle,
            center=self.center,
            scale=self.scale,
            border_value=self.pad_val,
            interpolation=self.interpolation,
        )
        results['img'] = img_rotated.astype(img.dtype)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(angle={self.angle}, '
        repr_str += f'center={self.center}, '
        repr_str += f'scale={self.scale}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'random_negative_prob={self.random_negative_prob}, '
        repr_str += f'interpolation={self.interpolation}{self.extra_repr()})'
        return repr_str
