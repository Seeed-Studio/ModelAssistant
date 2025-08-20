# Copyright (c) OpenMMLab. All rights reserved.
"""
Swift YOLO QAT Configuration Template

This is a template configuration file for Swift YOLO Quantization Aware Training.
Adapt this configuration based on your specific Swift YOLO implementation.

Usage:
    python tools/quantization.py configs/swift_yolo/swift_yolo_qat_template.py \
        path/to/swift_yolo_pretrained.pth \
        --work-dir work_dirs/swift_yolo_qat
"""

from mmengine.config import read_base

# TODO: Replace with your actual Swift YOLO base configuration
# with read_base():
#     from .swift_yolo_base import *

# For now, we provide a template - you need to adapt this based on your Swift YOLO config
from mmengine.optim import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import CosineAnnealingLR, ConstantLR
from mmengine.runner import EpochBasedTrainLoop

from sscma.datasets.transforms.loading import LoadImageFromFile, LoadAnnotations
from sscma.datasets.transforms.processing import RandomResize
from sscma.datasets.transforms.formatting import PackDetInputs
from sscma.datasets.transforms.transforms import (
    Pad,
    RandomCrop,
    RandomFlip,
    Resize,
    HSVRandomAug,
)
from sscma.engine.schedulers import QuadraticWarmupLR
from sscma.engine.hooks import QuantizerSwitchHook
from sscma.quantizer import SwiftYOLOQuantModel

# =============================================================================
# QAT Configuration Parameters
# =============================================================================
imgsz = (640, 640)  # Input image size - adapt based on your Swift YOLO config
dump_config = False

# Training configuration
max_epochs = 5
num_last_epochs = 2
base_lr = 0.00002

# =============================================================================
# Data Pipeline for QAT Training
# =============================================================================
train_pipeline = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend="pillow",
        backend_args=None,
    ),
    dict(type=LoadAnnotations, imdecode_backend="pillow", with_bbox=True),
    dict(type=HSVRandomAug),
    dict(
        type=RandomResize,
        scale=(imgsz[0] * 2, imgsz[1] * 2),
        ratio_range=(0.5, 1.5),
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs),
]

# =============================================================================
# Model Configuration for QAT
# =============================================================================

# TODO: Replace this with your actual Swift YOLO model configuration
# This is a placeholder - you need to define your Swift YOLO model here
model = dict(
    type='SwiftYOLO',  # Replace with your Swift YOLO class name
    # TODO: Add your backbone, neck, head configurations here
    # backbone=dict(...),
    # neck=dict(...),
    # bbox_head=dict(...),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None
    ),
    train_cfg=dict(
        # TODO: Add your training configuration
    ),
    test_cfg=dict(
        # TODO: Add your testing configuration
    )
)

# Update model head for QAT compatibility
# TODO: Uncomment and adapt these lines based on your model structure
# model['bbox_head'].update(train_cfg=model['train_cfg'])
# model['bbox_head'].update(test_cfg=model['test_cfg'])

# Configure the quantized model wrapper
quantizer_config = dict(
    type=SwiftYOLOQuantModel,
    # TODO: Update these based on your model structure
    # bbox_head=model['bbox_head'],
    # data_preprocessor=model['data_preprocessor'],
)

# =============================================================================
# Dataset Configuration
# =============================================================================

# TODO: Replace with your actual dataset configuration
train_dataloader = dict(
    batch_size=32,
    num_workers=16,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',  # Replace with your dataset type
        # TODO: Add your dataset configuration
        # data_root='path/to/your/data',
        # ann_file='annotations/train.json',
        # data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
        # filter_cfg=dict(filter_empty_gt=True, min_size=32),
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',  # Replace with your dataset type
        # TODO: Add your validation dataset configuration
        # data_root='path/to/your/data',
        # ann_file='annotations/val.json',
        # data_prefix=dict(img='val2017/'),
        test_mode=True,
        # pipeline=val_pipeline,
    )
)

# =============================================================================
# Training Configuration
# =============================================================================
train_cfg = dict(
    type=EpochBasedTrainLoop,
    max_epochs=max_epochs,
    val_interval=1,
    val_begin=1,
    dynamic_intervals=None,
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# =============================================================================
# Optimizer Configuration
# =============================================================================
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# =============================================================================
# Learning Rate Scheduler
# =============================================================================
param_scheduler = [
    dict(
        # Quadratic warmup for 1 epoch
        type=QuadraticWarmupLR,
        by_epoch=True,
        begin=0,
        end=1,
        convert_to_iter_based=True,
    ),
    dict(
        # Cosine annealing from epoch 1 to (max_epochs - num_last_epochs)
        type=CosineAnnealingLR,
        eta_min=base_lr * 0.05,
        begin=1,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        # Constant learning rate for the last epochs
        type=ConstantLR,
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    ),
]

# =============================================================================
# QAT Specific Hooks
# =============================================================================
custom_hooks = [
    dict(
        type=QuantizerSwitchHook,
        freeze_quantizer_epoch=max_epochs // 3,      # Freeze quantizer parameters at epoch 1
        freeze_bn_epoch=max_epochs // 3 * 2,         # Freeze batch norm statistics at epoch 3
    ),
]

# =============================================================================
# Runtime Configuration
# =============================================================================
default_scope = 'sscma'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer'
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

# =============================================================================
# Evaluation Configuration
# =============================================================================
val_evaluator = dict(
    type='CocoMetric',
    ann_file='path/to/your/val/annotations.json',  # TODO: Update path
    metric='bbox',
    format_only=False
)

test_evaluator = val_evaluator