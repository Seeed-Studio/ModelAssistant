# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './mobnetv2_0.35_rep_1bx16_300e_cifar10.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)


# ========================Suggested optional parameters========================
# MODEL
gray = True
widen_factor=0.35

# DATA
dataset_type = 'sscma.MNIST'
height = 32
width = 32
imgsz = (width, height)
data_root = 'datasets/'
train_ann = ''
train_data = 'mnist/'
val_ann = ''
val_data = 'mnist/'

# TRAIN
batch = 128
workers = 16
val_batch=batch
val_workers=workers
persistent_workers = True
# ================================END=================================
data_preprocessor = dict(
    type='sscma.ClsDataPreprocessor',
    mean=[0, 0, 0],
    std=[255.0, 255.0, 255.0],
    to_rgb=True,
)
model = dict(
    type='sscma.ImageClassifier',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0.0] if gray else [0.0, 0.0, 0.0],
        std=[255.0] if gray else [255.0, 255.0, 255.0],
    ),
    backbone=dict(type='MobileNetv2', gray_input=gray, widen_factor=widen_factor, out_indices=(2,), rep=True),
    neck=dict(type='sscma.GlobalAveragePooling'),
    head=dict(
        type='sscma.LinearClsHead',
        in_channels=32,
        loss=dict(type='sscma.CrossEntropyLoss', loss_weight=1.0),
    ),
)

train_pipeline = [
    dict(type='mmengine.Resize', scale=imgsz),
    dict(type='sscma.RandomRotate', angle=10.0, prob=0.5),
    dict(type='sscma.PackClsInputs'),
]

test_pipeline = [
    dict(type='mmengine.Resize', scale=imgsz),
    dict(type='sscma.PackClsInputs'),
]

train_dataloader = dict(
    # Training dataset configurations
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type, data_root=data_root, data_prefix=train_data, pipeline=train_pipeline, test_mode=False
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
    _delete_=True,
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
    _delete_=True,
)

test_dataloader = val_dataloader
