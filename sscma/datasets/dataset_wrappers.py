# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from typing import Union

from torch.utils.data.dataset import ConcatDataset as _ConcatDataset
from sscma.registry import DATASETS
from mmdet.datasets import CocoDataset


@DATASETS.register_module()
class SemiDataset(_ConcatDataset):
    """
    For merging real labeled and pseudo-labeled datasets in semi-supervised.

    Params:
        sup_dataset (dict): Real labeled dataset configuration
        unsup_dataset (dict): Pseudo-labeled dataset configuration
    """

    def __init__(self, sup_dataset: dict, unsup_dataset: dict, **kwargs) -> None:
        self._sup_dataset: CocoDataset = DATASETS.build(sup_dataset)
        self._unsup_dataset: CocoDataset = DATASETS.build(unsup_dataset)

        super(SemiDataset, self).__init__((self._sup_dataset, self._unsup_dataset))

        self.CLASSES: Union[list, tuple] = self.sup_dataset.METAINFO['classes']

    @property
    def sup_dataset(self):
        # get real labeled dataset
        return self._sup_dataset

    @property
    def unsup_dataset(self):
        # get pseudo labeled dataset
        return self._unsup_dataset
