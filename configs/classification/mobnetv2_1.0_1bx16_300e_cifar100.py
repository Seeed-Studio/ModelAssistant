# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './mobnetv2_1.0_1bx16_300e_cifar10.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================

# MODEL
num_classes = 100
widen_factor=1.0
# DATA
dataset_type = 'sscma.CIFAR100'
data_root = 'datasets/'
train_ann = ''
train_data = 'cifar100/'
val_ann = ''
val_data = 'cifar100/'
# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='sscma.MobileNetV2', widen_factor=widen_factor),
    neck=dict(type='sscma.GlobalAveragePooling'),
    head=dict(
        type='sscma.LinearClsHead',
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
