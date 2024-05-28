# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = '../_base_/default_runtime_det.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# MODEL
num_classes = 2
widen_factor = 1

# DATA
dataset_type = 'CustomFomoCocoDataset'
# datasets link: https://public.roboflow.com/object-detection/mask-wearing
data_root = 'https://public.roboflow.com/ds/o8GgfOIazi?key=hES8s8Gy7u'

train_ann = 'train/_annotations.coco.json'
train_data = 'train/'
val_ann = 'valid/_annotations.coco.json'
val_data = 'valid/'

height = 96
width = 96
imgsz = (width, height)

# TRAIN
batch = 16
workers = 1
persistent_workers = True

val_batch = 1
val_workers = 1

lr = 0.001
epochs = 100

weight_decay = 0.0005
momentum = (0.9, 0.99)

# ================================END=================================

default_hooks = dict(visualization=dict(type='mmdet.DetVisualizationHook', score_thr=0.8))

visualizer = dict(type='FomoLocalVisualizer', fomo=True)


data_preprocessor = dict(
    type='mmdet.DetDataPreprocessor', mean=[0, 0, 0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True, pad_size_divisor=32
)
model = dict(
    type='Fomo',
    data_preprocessor=data_preprocessor,
    backbone=dict(type='MobileNetV2', widen_factor=widen_factor, out_indices=(2,), rep=True),
    head=dict(
        type='FomoHead',
        input_channels=[64],
        num_classes=num_classes,
        middle_channel=48,
        act_cfg='ReLU6',
        loss_cls=dict(type='BCEWithLogitsLoss', reduction='none', pos_weight=40),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
    ),
    skip_preprocessor=True,
)


albu_train_transforms = [
    dict(type='Rotate', limit=30),
    dict(type='RandomBrightnessContrast', brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01),
    dict(type='HorizontalFlip', p=0.5),
]
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='mmdet.LoadAnnotations', with_bbox=True),
]

train_pipeline = [
    *pre_transform,
    dict(type='mmdet.Resize', scale=imgsz),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
    ),
    dict(type='Bbox2FomoMask', downsample_factor=(8,), num_classes=num_classes),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'fomo_mask',
            'img_path',
            'img_id',
            'instances',
            'img_shape',
            'ori_shape',
            'gt_bboxes',
            'gt_bboxes_labels',
        ),
    ),
]

test_pipeline = [
    *pre_transform,
    dict(type='mmdet.Resize', scale=imgsz),
    dict(type='Bbox2FomoMask', downsample_factor=(8,), num_classes=num_classes),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('fomo_mask', 'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'),
    ),
]

train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=train_data),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img=val_data),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

# data_preprocessor=dict(type='mmdet.DetDataPreprocessor')
# optimizer


find_unused_parameters = True

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=lr, betas=momentum, weight_decay=weight_decay, eps=1e-7),
)

# evaluator
val_evaluator = dict(type='FomoMetric')
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=epochs)

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
# cfg=dict(compile=True)
