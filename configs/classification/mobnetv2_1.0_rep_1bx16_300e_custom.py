# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './base.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# MODEL
num_classes = 3
gray = False
widen_factor = 1.0

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
# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[0.0] if gray else [0.0, 0.0, 0.0],
        std=[255.0] if gray else [255.0, 255.0, 255.0],
        bgr_to_rgb=False if gray else True,
    ),
    backbone=dict(type='MobileNetv2', widen_factor=widen_factor, rep=True, gray_input=gray),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        in_channels=128,
        num_classes=num_classes,
    ),
)

albu_train_transforms = [
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
    dict(type='Affine', translate_percent=[0.05, 0.30], p=0.3),
    # dict(type="ISONoise",),
    # dict(type="RandomFog"),
    # dict(type="RandomSunFlare"),
    # dict(type="RandomToneCurve"),
    dict(type='RGBShift'),
    # dict(type='Blur', p=0.3),
    dict(type='MedianBlur', blur_limit=3, p=0.5),
    dict(type='ToGray', p=0.3),
    dict(type='CLAHE', p=0.3),
    dict(type='HorizontalFlip'),
    dict(type='VerticalFlip'),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        keymap={'img': 'image'},
    ),
    dict(type='mmengine.Resize', scale=imgsz),
    # dict(type='mmcls.ColorJitter', brightness=0.3, contrast=0.3),
    dict(type='mmcls.Rotate', angle=30.0, prob=0.5),
    # dict(type='mmcls.RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='mmcls.PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmengine.Resize', scale=imgsz),
    dict(type='mmcls.PackClsInputs'),
]
if gray:
    train_pipeline.insert(-2, dict(type='Color2Gray', one_channel=True))
    test_pipeline.insert(-2, dict(type='Color2Gray', one_channel=True))

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
        pipeline=train_pipeline,
    ),
)

test_dataloader = val_dataloader
