_base_ = ['../_base_/default_runtime_det.py']
default_scope = 'mmyolo'


# ========================Suggested optional parameters========================
# DATA
# Types of datasets ,The type of the dataset, you can follow sscma/datasets/ to see the types we have defined,
# or you can use the types used by other mmlab libraries,
# but you need to prefix them with the appropriate prefixes to be on the safe side
dataset_type = 'sscma.CustomYOLOv5CocoDataset'
# Path to the dataset's root directory
# dataset link: https://universe.roboflow.com/team-roboflow/coco-128
data_root = 'https://universe.roboflow.com/ds/z5UOcgxZzD?key=bwx9LQUT0t'
# Path to the annotation file for the training set, both absolute and relative paths are acceptable,
# if it is a relative path, it must be relative to "data_root".
train_ann = 'train/_annotations.coco.json'
# Path to the training set data file, both absolute and relative, if relative, it must be relative to "data_root".
train_data = 'train/'
# Path to the validation set annotation file, both absolute and relative paths are acceptable,
# if it is a relative path, it must be a relative path to data_root.
val_ann = 'valid/_annotations.coco.json'
# Path to the validation set data file, both absolute and relative paths are allowed,
# if it is a relative path, it must be a relative path to data_root.
val_data = 'valid/'
# Height of the model input data
height = 640
# Width of the model input data
width = 640
# The width and height of the model input data
imgsz = (width, height)
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# MODEL
# Scaling factor for model depth
deepen_factor = 0.33
# Scaling factor for model width
widen_factor = 0.5
# Number of categories in the dataset
num_classes = 71

# TRAIN
# Learning rate of the model
lr = 0.01
# Total number of rounds of model training
epochs = 300
# Number of input data per iteration in the model training phase
batch = 64
# Number of threads used to load data during training, this value should be adjusted accordingly to the training batch
workers = 8
# Model weight saving interval in epochs
save_interval = 5
# Last epoch number to switch training pipeline
num_last_epochs = 15
# Learning rate scaling factor
lr_factor = 0.01
# Optimizer weight decay value
weight_decay = 0.0005
momentum=0.937

# VAL
# Number of input data per iteration in the model validation phase
val_batch = 1
# Number of threads used to load data during validation, this value should be adjusted accordingly to the validation batch
val_workers = workers
# Model validation interval in epoch
val_interval = 5
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# ================================END=================================

# Config of batch shapes. Only on val.
# It means not used if batch_shapes_cfg is None.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch,
    img_size=imgsz[0],
    size_divisor=32,
    extra_pad_ratio=0.5,
)

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio

# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ============================== Unmodified in most cases ===================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor', mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True
    ),
    backbone=dict(
        type='YOLOv6EfficientRep',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    neck=dict(
        type='YOLOv6RepPAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[128, 256, 512],
        num_csp_blocks=12,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    bbox_head=dict(
        type='YOLOv6Head',
        head_module=dict(
            type='YOLOv6HeadModule',
            num_classes=num_classes,
            in_channels=[128, 256, 512],
            widen_factor=widen_factor,
            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=[8, 16, 32],
        ),
        loss_bbox=dict(
            type='IoULoss', iou_mode='giou', bbox_format='xyxy', reduction='mean', loss_weight=2.5, return_iou=False
        ),
    ),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(
            type='BatchATSSAssigner', num_classes=num_classes, topk=9, iou_calculator=dict(type='mmdet.BboxOverlaps2D')
        ),
        assigner=dict(type='BatchTaskAlignedAssigner', num_classes=num_classes, topk=13, alpha=1, beta=6),
    ),
    test_cfg=dict(
        multi_label=True, nms_pre=30000, score_thr=0.001, nms=dict(type='nms', iou_threshold=0.65), max_per_img=300
    ),
)

# The training pipeline of YOLOv6 is basically the same as YOLOv5.
# The difference is that Mosaic and RandomAffine will be closed in the last 15 epochs. # noqa
pre_transform = [dict(type='LoadImageFromFile', backend_args=None), dict(type='LoadAnnotations', with_bbox=True)]

train_pipeline = [
    *pre_transform,
    dict(type='Mosaic', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # imgsz is (width, height)
        border=(-imgsz[0] // 2, -imgsz[1] // 2),
        border_val=(114, 114, 114),
        max_shear_degree=0.0,
    ),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction')
    ),
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=imgsz),
    dict(type='LetterResize', scale=imgsz, allow_scale_up=True, pad_val=dict(img=114)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_translate_ratio=0.1,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_shear_degree=0.0,
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
    collate_fn=dict(type='yolov5_collate'),
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
    dict(type='LoadImageFromFile', backend_args=None),
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

# Optimizer and learning rate scheduler of YOLOv6 are basically the same as YOLOv5. # noqa
# The difference is that the scheduler_type of YOLOv6 is cosine.
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=batch,
    ),
    constructor='YOLOv5OptimizerConstructor',
)

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook', scheduler_type='cosine', lr_factor=lr_factor, max_epochs=epochs
    ),
    checkpoint=dict(type='CheckpointHook', interval=save_interval, max_keep_ckpts=max_keep_ckpts, save_best='auto'),
)

custom_hooks = [
    dict(
        type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49
    ),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=epochs - num_last_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
]

val_evaluator = dict(type='mmdet.CocoMetric', proposal_nums=(100, 1, 10), ann_file=data_root + val_ann, metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=epochs,
    val_interval=val_interval,
    dynamic_intervals=[(epochs - num_last_epochs, 1)],
    _delete_=True,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
