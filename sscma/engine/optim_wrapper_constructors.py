"""This module is used to implement and register the custom optimizer wrapper
constructor.

Optimizer wrapper constructor is used to configure the optimize parameters,
such as learning rate, momentum in a custom way. You can customize it refer to https://github.com/open-mmlab/mmdetection/blob/3.x/mmdet/engine/optimizers/layer_decay_optimizer_constructor.py

The default implementation only does the register process. Users need to rename the
``CustomOptimWrapperConstructor`` to the real name of the optimizer wrapper
and implement constructor it.
"""  # noqa: E501

from mmengine.optim import DefaultOptimWrapperConstructor

from sscma.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class CustomOptimWrapperConstructor(DefaultOptimWrapperConstructor):
    ...
