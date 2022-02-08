"""This module is used to implement and register the custom optimizer wrapper.

``OptimWrapper`` is a functional (mixed-precision training,
gradient accumulation .etc.) wrapper on pytorch Optimizer, users do not need to
 implement ``OptimWrapper`` on their own in general.

The default implementation only does the register process. Users need to rename
the ``CustomOptimWrapper`` to the real name of the optimizer and implement it.
"""
from mmengine.optim import OptimWrapper

from sscma.registry import OPTIM_WRAPPERS


@OPTIM_WRAPPERS.register_module()
class CustomOptimWrapper(OptimWrapper):
    ...
