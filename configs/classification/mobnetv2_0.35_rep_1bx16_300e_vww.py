# Copyright (c) Seeed Tech Ltd. All rights reserved.
_base_ = './base.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)
# ========================Suggested optional parameters========================
# MODEL
num_classes = 2
gray = False
widen_factor = 0.5
# DATA
dataset_type = 'VWW'
data_root = 'datasets/vww/'
height = 96
width = 96
imgsz = (width, height)

train_ann = 'annotations/instances_train.json'
train_data = 'trainval/'

val_ann = 'annotations/instances_val.json'
val_data = 'trainval/'

# ================================END=================================


model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='MobileNetv2', widen_factor=widen_factor, rep=True, gray_input=gray, _delete_=True),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        in_channels=64,
        num_classes=num_classes,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
)

train_dataloader = dict(
    # Training dataset configurations
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=train_data,
        ann_file=train_ann,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    # Valid dataset configurations
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=val_data,
        ann_file=val_ann,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = val_dataloader
