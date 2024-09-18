# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, Dict, List, Optional, Union
import numpy as np

import mmengine
from mmengine.dataset import Compose
from sscma.datasets.transforms import BaseTransform
from sscma.datasets.transforms.utils import cache_randomness
from sscma.datasets.transforms.basetransform import BaseTransform

# Define type of transform or transform config
Transform = Union[Dict, Callable[[Dict], Dict]]


class RandomChoice(BaseTransform):
    """Process data with a randomly chosen transform from given candidates.

    Args:
        transforms (list[list]): A list of transform candidates, each is a
            sequence of transforms.
        prob (list[float], optional): The probabilities associated
            with each pipeline. The length should be equal to the pipeline
            number and the sum should be 1. If not given, a uniform
            distribution will be assumed.

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomChoice',
        >>>         transforms=[
        >>>             [dict(type='RandomHorizontalFlip')],  # subpipeline 1
        >>>             [dict(type='RandomRotate')],  # subpipeline 2
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(
        self,
        transforms: List[Union[Transform, List[Transform]]],
        prob: Optional[List[float]] = None,
    ):

        super().__init__()

        if prob is not None:
            assert mmengine.is_seq_of(prob, float)
            assert len(transforms) == len(prob), (
                "``transforms`` and ``prob`` must have same lengths. "
                f"Got {len(transforms)} vs {len(prob)}."
            )
            assert sum(prob) == 1

        self.prob = prob
        self.transforms = [Compose(transforms) for transforms in transforms]

    def __iter__(self):
        return iter(self.transforms)

    @cache_randomness
    def random_pipeline_index(self) -> int:
        """Return a random transform index."""
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        idx = self.random_pipeline_index()
        return self.transforms[idx](results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(transforms = {self.transforms}"
        repr_str += f"prob = {self.prob})"
        return repr_str
