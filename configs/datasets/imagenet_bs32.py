# Copyright (c) OpenMMLab. All rights reserved.
from sscma.datasets import LanceDataset
from sscma.datasets.transforms import (
    RandomFlip,
    RandomResizedCrop,
    RandomFlip,
    ResizeEdge,
    CenterCrop,
    PackInputs,
    RandomResizedCrop,
    ResizeEdge,
    CenterCrop,
)
from mmengine.dataset import DefaultSampler
from sscma.evaluation import Accuracy

# dataset settings
dataset_type = LanceDataset

data_root='datasets/imagenet'

train_pipeline = [
    # dict(type=LoadImageFromFile,imdecode_backend='cv2'),
    dict(type=RandomResizedCrop, scale=224),
    dict(type=RandomFlip, prob=0.5, direction="horizontal"),
    dict(type=PackInputs),
]

test_pipeline = [
    # dict(type=LoadImageFromFile,imdecode_backend='cv2'),
    dict(type=ResizeEdge, scale=256, edge="short"),
    dict(type=CenterCrop, crop_size=224),
    dict(type=PackInputs),
]

train_dataloader = dict(
    batch_size=32,
    num_workers=2,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="train"),
        pipeline=train_pipeline,
    ),
    sampler=dict(type=DefaultSampler, shuffle=True),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    pin_memory=True,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="valid"),
        pipeline=test_pipeline,
    ),
    sampler=dict(type=DefaultSampler, shuffle=False),
)
val_evaluator = dict(type=Accuracy, topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
