# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './base.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# MODEL
num_classes = 3
widen_factor = 1.0

# DATA
dataset_type = 'sscma.CustomDataset'
# datasets link: https://public.roboflow.com/classification/rock-paper-scissors
data_root = 'https://public.roboflow.com/ds/dTMAyuzrmY?key=VbTbUwLEYG'
train_data = 'train/'
val_data = 'valid/'
train_ann = ''
val_ann = ''

height = 96
width = 96
imgsz = (width, height)
# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='sscma.MobileNetV2', widen_factor=widen_factor),
    neck=dict(type='sscma.GlobalAveragePooling'),
    head=dict(
        type='sscma.LinearClsHead',
        in_channels=1280,
        num_classes=num_classes,
        loss=dict(type='sscma.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5) if num_classes > 5 else 1,
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmengine.Resize', scale=imgsz),
    dict(type='sscma.ColorJitter', brightness=0.3, contrast=0.2),
    dict(type='sscma.RandomRotate', angle=30.0, prob=0.6),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='sscma.PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmengine.Resize', scale=imgsz),
    dict(type='sscma.PackClsInputs'),
]

train_dataloader = dict(
    # Training dataset configurations
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=train_data,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    # Valid dataset configurations
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=val_data,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader
