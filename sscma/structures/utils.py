# copyright Copyright (c) Seeed Technology Co.,Ltd.
from typing import List

from mmengine.structures import InstanceData

from sscma.structures import PoseDataSample


def merge_data_samples(data_samples: List[PoseDataSample]) -> PoseDataSample:
    assert isinstance(data_samples, List), f'data_samples should be a list instead of {type(data_samples)}'
    assert isinstance(
        data_samples[0], PoseDataSample
    ), f'data_samples should be PoseDataSample instead of {type(data_samples[0])}'

    merged = PoseDataSample(metainfo=data_samples[0].metainfo)

    if 'gt_instances' in data_samples[0]:
        merged.gt_instances = InstanceData.cat([d.gt_instances for d in data_samples])

    if 'pred_instances' in data_samples[0]:
        merged.pred_instances = InstanceData.cat([d.pred_instances for d in data_samples])

    return merged
