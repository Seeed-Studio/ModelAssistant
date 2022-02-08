"""This module is used to implement and register the custom optimizer.

If you want to use a custom optimizer which has not been implemented by
``torch.optim``, you can customize your optimizer here.

The default implementation only does the register process. Users need to rename
the ``CustomOptimizer`` to the real name of the optimizer and implement it.
"""

from torch.optim import Optimizer

from sscma.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class CustomOptimizer(Optimizer):
    ...
