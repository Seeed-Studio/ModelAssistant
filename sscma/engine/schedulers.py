"""This module is used to implement and register the custom scheduler.

MMEngine has provided rich parameter scheduler in https://mmengine.readthedocs.io/en/latest/api/optim.html.
If these schedulers cannot meet your requirements, you can customize your
scheduer here.

The default implementation only does the register process. Users need to rename
the ``CustomXXXcheduler`` to the real name of the scheduler and implement it.
"""  # noqa: E501
from mmengine.optim import _ParamScheduler
from mmengine.optim.scheduler.lr_scheduler import LRSchedulerMixin
from mmengine.optim.scheduler.momentum_scheduler import MomentumSchedulerMixin

from sscma.registry import PARAM_SCHEDULERS


@PARAM_SCHEDULERS.register_module()
class CustomParameterScheduler(_ParamScheduler):
    ...


@PARAM_SCHEDULERS.register_module()
class CustomMomentumScheduler(MomentumSchedulerMixin,
                              CustomParameterScheduler):
    ...


@PARAM_SCHEDULERS.register_module()
class CustomLRScheduler(LRSchedulerMixin, CustomParameterScheduler):
    ...
