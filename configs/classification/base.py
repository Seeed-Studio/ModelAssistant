# Copyright (c) Seeed Tech Ltd. All rights reserved.
_base_ = '../_base_/default_runtime_cls.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# MODEL
num_classes = 3
widen_factor = 0.5

gray = False
# DATA
dataset_type = 'mmcls.CustomDataset'
# datasets link: https://public.roboflow.com/classification/rock-paper-scissors
data_root = 'https://public.roboflow.com/ds/dTMAyuzrmY?key=VbTbUwLEYG'
train_data = 'train/'
val_data = 'valid/'
train_ann = ''
val_ann = ''


height = 192
width = 192
imgsz = (width, height)

# TRAIN
batch = 128
workers = 16

val_batch = batch
val_workers = workers
persistent_workers = True
lr = 0.01
epochs = 100
weight_decay = 0.0001
momentum = 0.9

# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0.0] if gray else [0.0, 0.0, 0.0],
        std=[255.0] if gray else [255.0, 255.0, 255.0],
    ),
    backbone=dict(type='MobileNetv2', widen_factor=widen_factor),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        in_channels=32,
        num_classes=num_classes,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5) if num_classes > 5 else 1,
    ),
)

albu_train_transforms = [
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
    dict(type='Affine', translate_percent=[0.05, 0.30], p=0.3),
    dict(type='RandomToneCurve'),
    dict(type='MedianBlur', blur_limit=3, p=0.5),
    dict(type='ToGray', p=0.3),
    dict(type='CLAHE', p=0.3),
    dict(type='HorizontalFlip'),
    dict(type='VerticalFlip'),
    dict(type='RGBShift'),
]


train_pipeline = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        keymap={'img': 'image'},
    ),
    dict(type='mmengine.Resize', scale=imgsz),
    dict(type='mmcls.Rotate', angle=30.0, prob=0.5),
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='mmengine.Resize', scale=imgsz),
    dict(type='mmcls.PackClsInputs'),
]
if gray:
    train_pipeline.insert(-2, dict(type='Color2Gray', one_channel=True))
    test_pipeline.insert(-2, dict(type='Color2Gray', one_channel=True))
    albu_train_transforms.pop()

train_dataloader = dict(
    # Training dataset configurations
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=train_data,
        pipeline=train_pipeline,
        test_mode=False,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    # Valid dataset configurations
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=val_data,
        pipeline=test_pipeline,
        test_mode=True,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = val_dataloader

# evaluator
val_evaluator = dict(type='mmcls.Accuracy', topk=(1, 5) if num_classes > 5 else 1)
test_evaluator = val_evaluator


val_cfg = dict()
test_cfg = dict()

# optimizer
optim_wrapper = dict(optimizer=dict(type='SGD', lr=lr, momentum=momentum, weight_decay=weight_decay))
# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=30, start_factor=0.001, by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=1,
        end=500,
        milestones=[100, 200, 250],
        gamma=0.1,
        by_epoch=True,
    ),
]

auto_scale_lr = dict(base_batch_size=batch)

train_cfg = dict(by_epoch=True, max_epochs=epochs)
