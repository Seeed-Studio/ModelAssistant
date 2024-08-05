# Copyright (c) OpenMMLab. All rights reserved.

from .cross_entropy_loss import CrossEntropyLoss, binary_cross_entropy, cross_entropy

__all__ = [
    "cross_entropy",
    "binary_cross_entropy",
    "CrossEntropyLoss",
]
