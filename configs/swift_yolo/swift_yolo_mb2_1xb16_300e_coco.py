# Copyright (c) Seeed Tech Ltd. All rights reserved.
_base_ = [
    '../_base_/default_runtime_det.py',
]

default_scope = 'mmyolo'
# ========================Suggested optional parameters========================
# MODEL
# The scaling factor that controls the depth of the network structure
deepen_factor = 1
# The scaling factor that controls the width of the network structure
widen_factor = 0.35
# Number of classes for classification
num_classes = 1

# DATA
# Dataset type, this will be used to define the dataset
dataset_type = 'sscma.CustomYOLOv5CocoDataset'
# Root path of data
# dataset link: https://universe.roboflow.com/team-roboflow/coco-128
data_root = 'https://universe.roboflow.com/ds/z5UOcgxZzD?key=bwx9LQUT0t'
# Path of train annotation file
train_ann = 'train/_annotations.coco.json'
# Prefix of train image path
train_data = 'train/'
# Path of val annotation file
val_ann = 'valid/_annotations.coco.json'
# Prefix of val image path
val_data = 'valid/'
height = 192
width = 192
imgsz = (width, height)  # width, height

# TRAIN
# batch
batch = 8
# workers
workers = 2
# Batch size of a single GPU during validation
val_batch = 1
# Worker to pre-fetch data for each single GPU during validation
val_workers = 1
persistent_workers = True

# ================================END=================================

randomness = dict(seed=6699)
# -----model related-----
# Basic size of multi-scale prior box
anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)],  # P5/32
]

# -----train val related-----
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
base_lr = 0.01
epochs = 300  # Maximum training epochs

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.5),  # NMS type and threshold
    max_per_img=300,
)  # Max number of detections of each image

# -----data related-----

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch,
    img_size=imgsz[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5,
)

# -----model related-----

# Strides of multi-scale prior box
strides = [8, 16, 32]
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
prior_match_thr = 4.0  # Priori box matching threshold
# The obj loss weights of the three output layers
obj_level_weights = [4.0, 1.0, 0.4]
lr_factor = 0.01  # Learning rate scaling factor
weight_decay = 0.0005
# Save model checkpoint and validation intervals
save_checkpoint_intervals = 5
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# model arch
model = dict(
    type='sscma.YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor', mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True
    ),
    backbone=dict(
        type='MobileNetv2',
        widen_factor=widen_factor,
        out_indices=(
            2,
            3,
            5,
        ),
        rep=True,
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[32, 48, 112],
        out_channels=[32, 48, 112],
        num_csp_blocks=1,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    bbox_head=dict(
        type='sscma.YOLOV5Head',
        head_module=dict(
            type='sscma.DetHead',
            num_classes=num_classes,
            in_channels=[32, 48, 112],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3,
        ),
        prior_generator=dict(type='mmdet.YOLOAnchorGenerator', base_sizes=anchors, strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=loss_bbox_weight,
            return_iou=True,
        ),
        loss_obj=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=loss_obj_weight),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights,
    ),
    test_cfg=model_test_cfg,
)

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01),
]

pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
]

train_pipeline = [
    *pre_transform,
    dict(type='Mosaic', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # imgsz is (width, height)
        border=(-imgsz[0] // 2, -imgsz[1] // 2),
        border_val=(114, 114, 114),
    ),
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={'img': 'image', 'gt_bboxes': 'bboxes'},
    ),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction')
    ),
]

train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=train_data),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    ),
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='YOLOv5KeepRatioResize', scale=imgsz),
    dict(type='LetterResize', scale=imgsz, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param'),
    ),
]

val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data),
        ann_file=val_ann,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg,
    ),
)

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=weight_decay,
        # nesterov=True,
        batch_size_per_gpu=batch,
    ),
    constructor='YOLOv5OptimizerConstructor',
)

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook', scheduler_type='linear', lr_factor=lr_factor, max_epochs=epochs
    ),
    checkpoint=dict(
        type='CheckpointHook', interval=save_checkpoint_intervals, save_best='auto', max_keep_ckpts=max_keep_ckpts
    ),
)

custom_hooks = [
    dict(
        type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49
    )
]

val_evaluator = dict(type='mmdet.CocoMetric', proposal_nums=(100, 1, 10), ann_file=data_root + val_ann, metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=save_checkpoint_intervals, _delete_=True)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
