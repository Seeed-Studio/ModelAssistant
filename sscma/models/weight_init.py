"""This module is used to implement and register the custom initializer.

MMEngine has implemented varies of initlizers in https://github.com/open-mmlab/mmengine/blob/main/mmengine/model/weight_init.py.
Users can customize the initializer in this file.

The default implementation only does the register process. Users need to rename
the ``CustomInitializer`` to the real name of the initializer and implement it.
"""  # noqa: E501

from mmengine.model.weight_init import BaseInit


class CustomInitializer(BaseInit): ...
