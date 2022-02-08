"""This module is used to implement and register the custom datasets.

If OpenMMLab series repositries have supported the target dataset, for example,
CocoDataset. You can simply use it by setting ``type=mmdet.CocoDataset`` in the
config file.

If you want to do some small modifications to the existing dataset,
you can inherit from it and override its methods:

Examples:
    >>> from mmdet.datasets import CocoDataset as MMDetCocoDataset
    >>>
    >>> class CocoDataset(MMDetCocoDataset):
    >>>     def load_data_list(self):
    >>>         ...

Don't worry about the duplicated name of the custom ``CocoDataset`` and the
mmdet ``CocoDataset``, they are registered into different registry nodes.

The default implementation only does the register process. Users need to rename
the ``CustomDataset`` to the real name of the target dataset, for example,
``WiderFaceDataset``, and then implement it.
"""

from mmengine.dataset import BaseDataset

from sscma.registry import DATASETS


@DATASETS.register_module()
class CustomDataset(BaseDataset):
    ...
