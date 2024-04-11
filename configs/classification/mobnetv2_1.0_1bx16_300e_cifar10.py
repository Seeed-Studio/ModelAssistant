# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './base.py'
# ========================Suggested optional parameters========================
# MODEL
num_classes = 10
widen_factor = 1.0

# DATA
dataset_type = 'mmcls.CIFAR10'
data_root = 'datasets/'
train_ann = ''
train_data = 'cifar10/'
val_ann = ''
val_data = 'cifar10/'
# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='mmcls.MobileNetV2', widen_factor=widen_factor),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        in_channels=1280,
        num_classes=num_classes,
    ),
)
train_dataloader = dict(
    # Training dataset configurations
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=train_data,
        test_mode=False,
    ),
)

val_dataloader = dict(
    # Valid dataset configurations
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=val_data,
        test_mode=True,
    ),
)

test_dataloader = val_dataloader
