# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union
from mmcv.transforms import BaseTransform, Compose
import copy
from sscma.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MutiBranchPipe(BaseTransform):
    def __init__(self, branch_field, piece_key: str = None, **branch_pipelines) -> None:
        self.branch_field = branch_field
        self.branch_pipelines = {branch: Compose(pipeline) for branch, pipeline in branch_pipelines.items()}
        self.piece_key = piece_key

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        multi_results = {}
        for branch in self.branch_field:
            multi_results[branch] = {'inputs': None, 'data_samples': None}
        for branch, pipeline in self.branch_pipelines.items():
            branch_results = pipeline(copy.deepcopy(results))
            if branch == self.piece_key:
                results['img'] = branch_results['inputs'].permute(1, 2, 0).cpu().numpy()
            # If one branch pipeline returns None,
            # it will sample another data from dataset.
            if branch_results is None:
                return None
            multi_results[branch] = branch_results

        format_results = {}
        for branch, results in multi_results.items():
            for key in results.keys():
                if format_results.get(key, None) is None:
                    format_results[key] = {branch: results[key]}
                else:
                    format_results[key][branch] = results[key]
        return format_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(branch_pipelines={list(self.branch_pipelines.keys())})'
        return repr_str
