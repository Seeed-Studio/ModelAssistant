"""This module is used to implement and register the custom model wrapper.

Since ``mmengine.runner.Runner`` will call ``train_step``, ``val_step`` and
``test_step`` in different phases. There should be a wrapper on
``DistributedDataParallel``, ``FullyShardedDataParallel`` .etc. to implements
these methods. MMEngine has provided the commonly used wrappers for users, but
you can still customize the wrapper for some special requirements.

The default implementation only does the register process. Users need to rename
the ``CustomWrapper`` to the real name of the wrapper and implement it.
"""
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# Define type of transform or transform config
Transform = Union[Dict, Callable[[Dict], Dict]]
from mmengine.registry import TRANSFORMS


from ..datasets.transforms.basetransform import BaseTransform

class Compose(BaseTransform):
    """Compose multiple transforms sequentially.

    Args:
        transforms (list[dict | callable]): Sequence of transform object or
            config dict to be composed.

    Examples:
        >>> pipeline = [
        >>>     dict(type='Compose',
        >>>         transforms=[
        >>>             dict(type='LoadImageFromFile'),
        >>>             dict(type='Normalize')
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self, transforms: Union[Transform, Sequence[Transform]]):
        super().__init__()

        if not isinstance(transforms, Sequence):
            transforms = [transforms]
        self.transforms: List = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict, but got'
                                f' {type(transform)}')

    def __iter__(self):
        """Allow easy iteration over the transform sequence."""
        return iter(self.transforms)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Call function to apply transforms sequentially.

        Args:
            results (dict): A result dict contains the results to transform.

        Returns:
            dict or None: Transformed results.
        """
        for t in self.transforms:
            results = t(results)  # type: ignore
            if results is None:
                return None
        return results

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string
