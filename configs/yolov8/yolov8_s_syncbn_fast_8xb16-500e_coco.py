_base_ = ['../_base_/default_runtime_det.py']
default_scope = 'mmyolo'


# ========================Suggested optional parameters========================
# DATA
# Types of datasets ,The type of the dataset, you can follow sscma/datasets/ to see the types we have defined,
# or you can use the types used by other mmlab libraries,
# but you need to prefix them with the appropriate prefixes to be on the safe side
dataset_type = 'sscma.CustomYOLOv5CocoDataset'
# Path to the dataset's root directory
data_root = 'data/coco/'
# Path to the annotation file for the training set, both absolute and relative paths are acceptable,
# if it is a relative path, it must be relative to "data_root".
train_ann = 'annotations/instances_train2017.json'
# Path to the training set data file, both absolute and relative, if relative, it must be relative to "data_root".
train_data = 'train2017/'
# Path to the validation set annotation file, both absolute and relative paths are acceptable,
# if it is a relative path, it must be a relative path to data_root.
val_ann = 'annotations/instances_val2017.json'
# Path to the validation set data file, both absolute and relative paths are allowed,
# if it is a relative path, it must be a relative path to data_root.
val_data = 'val2017/'
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
num_classes = 80

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
# Disable mosaic augmentation for final 10 epochs (stage 2)
close_mosaic_epochs = 10

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

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.7),  # NMS type and threshold
    max_per_img=300,
)  # Max number of detections of each image
# Config of batch shapes. Only on val.
# We tested YOLOv8-m will get 0.02 higher than not using it.
batch_shapes_cfg = None
# You can turn on `batch_shapes_cfg` by uncommenting the following lines.
# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',
#     batch_size=val_batch_size_per_gpu,
#     img_size=imgsz[0],
#     # The image scale of padding should be divided by pad_size_divisor
#     size_divisor=32,
#     # Additional paddings for pixel scale
#     extra_pad_ratio=0.5)

# Strides of multi-scale prior box
strides = [8, 16, 32]
# The output channel of the last stage
last_stage_out_channels = 1024
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100
tal_topk = 10  # Number of bbox selected in each level
tal_alpha = 0.5  # A Hyper-parameter related to alignment_metrics
tal_beta = 6.0  # A Hyper-parameter related to alignment_metrics
# TODO: Automatically scale loss_weight based on number of detection layers
loss_cls_weight = 0.5
loss_bbox_weight = 7.5
# Since the dfloss is implemented differently in the official
# and mmdet, we're going to divide loss_weight by 4.
loss_dfl_weight = 1.5 / 4


# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor', mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True
    ),
    backbone=dict(
        type='YOLOv8CSPDarknet',
        arch='P5',
        last_stage_out_channels=last_stage_out_channels,
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    neck=dict(
        type='YOLOv8PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='YOLOv8Head',
        head_module=dict(
            type='YOLOv8HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, last_stage_out_channels],
            widen_factor=widen_factor,
            reg_max=16,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='SiLU', inplace=True),
            featmap_strides=strides,
        ),
        prior_generator=dict(type='mmdet.MlvlPointGenerator', offset=0.5, strides=strides),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # scaled based on number of detection layers
        loss_cls=dict(type='mmdet.CrossEntropyLoss', use_sigmoid=True, reduction='none', loss_weight=loss_cls_weight),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=loss_bbox_weight,
            return_iou=False,
        ),
        loss_dfl=dict(type='mmdet.DistributionFocalLoss', reduction='mean', loss_weight=loss_dfl_weight),
    ),
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_classes,
            use_ciou=True,
            topk=tal_topk,
            alpha=tal_alpha,
            beta=tal_beta,
            eps=1e-9,
        )
    ),
    test_cfg=model_test_cfg,
)

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01),
]

pre_transform = [dict(type='LoadImageFromFile', backend_args=None), dict(type='LoadAnnotations', with_bbox=True)]

last_transform = [
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

train_pipeline = [
    *pre_transform,
    dict(type='Mosaic', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        # imgsz is (width, height)
        border=(-imgsz[0] // 2, -imgsz[1] // 2),
        border_val=(114, 114, 114),
    ),
    *last_transform,
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=imgsz),
    dict(type='LetterResize', scale=imgsz, allow_scale_up=True, pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114),
    ),
    *last_transform,
]

train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
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

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=10.0),
    optimizer=dict(
        type='SGD',
        lr=lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=batch,
    ),
    constructor='YOLOv5OptimizerConstructor',
)

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook', scheduler_type='linear', lr_factor=lr_factor, max_epochs=epochs
    ),
    checkpoint=dict(type='CheckpointHook', interval=save_interval, save_best='auto', max_keep_ckpts=max_keep_ckpts),
)

custom_hooks = [
    dict(
        type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49
    ),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
]

val_evaluator = dict(type='mmdet.CocoMetric', proposal_nums=(100, 1, 10), ann_file=data_root + val_ann, metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=epochs,
    val_interval=save_interval,
    dynamic_intervals=[((epochs - close_mosaic_epochs), val_interval)],
    _delete_=True,
)


val_cfg = dict()
test_cfg = dict()
