_base_ = './base.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# model settings
num_classes = 10

# dataset settings
dataset_type = 'mmcls.CustomDataset'
data_root = 'datasets/'

height = 96
width = 96
imgsz = (width, height)

train_ann = ''
train_data = 'train/'
val_ann = ''
val_data = 'valid/'
# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='mmcls.MobileNetV2', widen_factor=1.0),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        in_channels=1280,
        num_classes=num_classes,
        loss=dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmengine.Resize', scale=(height, width)),
    dict(type='mmcls.ColorJitter', brightness=0.3, contrast=0.2),
    dict(type='mmcls.Rotate', angle=30.0, prob=0.6),
    dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmengine.Resize', scale=(height, width)),
    dict(type='mmcls.PackClsInputs'),
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
