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
save_interval = 1
# Last epoch number to switch training pipeline
num_last_epochs = 15
# Learning rate scaling factor
lr_factor = 0.01
# Optimizer weight decay value
weight_decay = 0.0005

# VAL
# Number of input data per iteration in the model validation phase
val_batch = 1
# Number of threads used to load data during validation, this value should be adjusted accordingly to the validation batch
val_workers = workers
# Model validation interval in epoch
val_interval = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 1
# ================================END=================================
# -----model related-----
# Basic size of multi-scale prior box
anchors = [
    [(12, 16), (19, 36), (40, 28)],  # P3/8
    [(36, 75), (76, 55), (72, 146)],  # P4/16
    [(142, 110), (192, 243), (459, 401)],  # P5/32
]


num_epoch_stage2 = 30  # The last 30 epochs switch evaluation interval
val_interval_stage2 = 1  # Evaluation interval
# ========================Possible modified parameters========================


model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS.
    nms_pre=30000,
    score_thr=0.001,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300,
)  # Max number of detections of each image


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
strides = [8, 16, 32]  # Strides of multi-scale prior box
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)

# Data augmentation
max_translate_ratio = 0.2  # YOLOv5RandomAffine
scaling_ratio_range = (0.1, 2.0)  # YOLOv5RandomAffine
mixup_prob = 0.15  # YOLOv5MixUp
randchoice_mosaic_prob = [0.8, 0.2]
mixup_alpha = 8.0  # YOLOv5MixUp
mixup_beta = 8.0  # YOLOv5MixUp

# -----train val related-----
loss_cls_weight = 0.3
loss_bbox_weight = 0.05
loss_obj_weight = 0.7
# BatchYOLOv7Assigner params
simota_candidate_topk = 10
simota_iou_weight = 3.0
simota_cls_weight = 1.0
prior_match_thr = 4.0  # Priori box matching threshold
obj_level_weights = [4.0, 1.0, 0.4]  # The obj loss weights of the three output layers

# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# ===============================Unmodified in most cases====================
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor', mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True
    ),
    backbone=dict(type='YOLOv7Backbone', arch='L', norm_cfg=norm_cfg, act_cfg=dict(type='SiLU', inplace=True)),
    neck=dict(
        type='YOLOv7PAFPN',
        block_cfg=dict(type='ELANBlock', middle_ratio=0.5, block_ratio=0.25, num_blocks=4, num_convs_in_block=1),
        upsample_feats_cat_first=False,
        in_channels=[512, 1024, 1024],
        # The real output channel will be multiplied by 2
        out_channels=[128, 256, 512],
        norm_cfg=norm_cfg,
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='YOLOv7Head',
        head_module=dict(
            type='YOLOv7HeadModule',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            featmap_strides=strides,
            num_base_priors=3,
        ),
        prior_generator=dict(type='mmdet.YOLOAnchorGenerator', base_sizes=anchors, strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight * (num_classes / 80 * 3 / num_det_layers),
        ),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True,
        ),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight * ((imgsz[0] / 640) ** 2 * 3 / num_det_layers),
        ),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights,
        # BatchYOLOv7Assigner params
        simota_candidate_topk=simota_candidate_topk,
        simota_iou_weight=simota_iou_weight,
        simota_cls_weight=simota_cls_weight,
    ),
    test_cfg=model_test_cfg,
)

pre_transform = [dict(type='LoadImageFromFile', backend_args=None), dict(type='LoadAnnotations', with_bbox=True)]

mosiac4_pipeline = [
    dict(type='Mosaic', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # note
        scaling_ratio_range=scaling_ratio_range,  # note
        # imgsz is (width, height)
        border=(-imgsz[0] // 2, -imgsz[1] // 2),
        border_val=(114, 114, 114),
    ),
]

mosiac9_pipeline = [
    dict(type='Mosaic9', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # note
        scaling_ratio_range=scaling_ratio_range,  # note
        # imgsz is (width, height)
        border=(-imgsz[0] // 2, -imgsz[1] // 2),
        border_val=(114, 114, 114),
    ),
]

randchoice_mosaic_pipeline = dict(
    type='RandomChoice', transforms=[mosiac4_pipeline, mosiac9_pipeline], prob=randchoice_mosaic_prob
)

train_pipeline = [
    *pre_transform,
    randchoice_mosaic_pipeline,
    dict(
        type='YOLOv5MixUp',
        alpha=mixup_alpha,  # note
        beta=mixup_beta,  # note
        prob=mixup_prob,
        pre_transform=[*pre_transform, randchoice_mosaic_pipeline],
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
    collate_fn=dict(type='yolov5_collate'),  # FASTER
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
    optimizer=dict(
        type='SGD',
        lr=lr,
        momentum=0.937,
        weight_decay=weight_decay,
        nesterov=True,
        batch_size_per_gpu=batch,
    ),
    constructor='YOLOv7OptimWrapperConstructor',
)

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook', scheduler_type='cosine', lr_factor=lr_factor, max_epochs=epochs  # note
    ),
    checkpoint=dict(
        type='CheckpointHook',
        save_param_scheduler=False,
        interval=save_interval,
        save_best='auto',
        max_keep_ckpts=max_keep_ckpts,
    ),
)

custom_hooks = [
    dict(
        type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49
    )
]

val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),  # Can be accelerated
    ann_file=data_root + val_ann,
    metric='bbox',
)
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=epochs,
    val_interval=val_interval,
    dynamic_intervals=[(epochs - num_epoch_stage2, val_interval_stage2)],
    _delete_=True,
)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
