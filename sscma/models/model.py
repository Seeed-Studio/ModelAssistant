"""This module is used to implement and register the custom model.

Follow the `guide <https://mmengine.readthedocs.io/en/latest/tutorials/model.html>`
in MMEngine to implement CustomModel

The default implementation only does the register process. Users need to rename
the ``CustomModel`` to the real name of the model and implement it.
"""  # noqa: E501
from mmengine.model import BaseModel

from sscma.registry import MODELS


@MODELS.register_module()
class CustomModel(BaseModel):
    ...
