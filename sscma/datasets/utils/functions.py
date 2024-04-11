# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import torch
from mmdet.structures import DetDataSample
from mmengine.registry import FUNCTIONS


@FUNCTIONS.register_module()
def fomo_collate(batch):
    img, label = [x['inputs'] for x in batch], [y['data_samples'] for y in batch]
    for i, label in enumerate(label):
        if label.shape[0] > 0:
            label[:, 0] = i
    return dict(inputs=torch.stack(img), data_samples=[DetDataSample(labels=torch.cat(label, 0))])
