from typing import List, Optional, Union
from mmcls.datasets.base_dataset import BaseDataset
import pyvww as pvw
import numpy as np
import copy
from sscma.registry import DATASETS


@DATASETS.register_module()
class VWW(BaseDataset):
    def __init__(
        self,
        ann_file: str = '',
        metainfo: Optional[dict] = None,
        data_root: str = '',
        data_prefix: Union[str, dict] = '',
        test_mode: bool = False,
        **kwargs,
    ):
        super().__init__(
            ann_file=ann_file,
            metainfo=metainfo,
            data_root=data_root,
            data_prefix=dict(root=data_prefix),
            test_mode=test_mode,
            **kwargs,
        )
        self._metainfo = {"classes": ["Nobody", "Somebody"]}

    def load_data_list(self) -> List[dict]:
        self.data_testt = pvw.pyvww.pytorch.VisualWakeWordsClassification(
            root=self.data_prefix['root'], annFile=self.ann_file
        )
        return copy.copy(self.data_testt.ids)

    def get_data_info(self, idx):
        img, label = self.data_testt[idx]
        img = np.asarray(img)
        return {"img": img, "gt_label": label, "sample_idx": idx}
