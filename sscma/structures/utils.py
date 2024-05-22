# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
from typing import List

from mmengine.structures import InstanceData
from mmengine.structures import LabelData

import torch
import torch.nn.functional as F

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


if hasattr(torch, 'tensor_split'):
    tensor_split = torch.tensor_split
else:
    # A simple implementation of `tensor_split`.
    def tensor_split(input: torch.Tensor, indices: list):
        outs = []
        for start, end in zip([0] + indices, indices + [input.size(0)]):
            outs.append(input[start:end])
        return outs


def cat_batch_labels(elements: List[LabelData], device=None):
    """Concat the ``label`` of a batch of :obj:`LabelData` to a tensor.

    Args:
        elements (List[LabelData]): A batch of :obj`LabelData`.
        device (torch.device, optional): The output device of the batch label.
            Defaults to None.

    Returns:
        Tuple[torch.Tensor, List[int]]: The first item is the concated label
        tensor, and the second item is the split indices of every sample.
    """
    item = elements[0]
    if 'label' not in item._data_fields:
        return None, None

    labels = []
    splits = [0]
    for element in elements:
        labels.append(element.label)
        splits.append(splits[-1] + element.label.size(0))
    batch_label = torch.cat(labels)
    if device is not None:
        batch_label = batch_label.to(device=device)
    return batch_label, splits[1:-1]


def batch_label_to_onehot(batch_label, split_indices, num_classes):
    """Convert a concated label tensor to onehot format.

    Args:
        batch_label (torch.Tensor): A concated label tensor from multiple
            samples.
        split_indices (List[int]): The split indices of every sample.
        num_classes (int): The number of classes.

    Returns:
        torch.Tensor: The onehot format label tensor.

    """
    sparse_onehot_list = F.one_hot(batch_label, num_classes)
    onehot_list = [
        sparse_onehot.sum(0)
        for sparse_onehot in tensor_split(sparse_onehot_list, split_indices)
    ]
    return torch.stack(onehot_list)


def stack_batch_scores(elements, device=None):
    """Stack the ``score`` of a batch of :obj:`LabelData` to a tensor.

    Args:
        elements (List[LabelData]): A batch of :obj`LabelData`.
        device (torch.device, optional): The output device of the batch label.
            Defaults to None.

    Returns:
        torch.Tensor: The stacked score tensor.
    """
    item = elements[0]
    if 'score' not in item._data_fields:
        return None

    batch_score = torch.stack([element.score for element in elements])
    if device is not None:
        batch_score = batch_score.to(device)
    return batch_score
