_base_ = [
    '../_base_/default_runtime_det.py',
]  # model settings

default_scope = 'mmdet'

# ========================Suggested optional parameters========================
# MODEL
num_classes = 71

# TRAIN
# dataset settings
dataset_type = 'sscma.CustomCocoDataset'
# dataset link: https://universe.roboflow.com/team-roboflow/coco-128
data_root = 'https://universe.roboflow.com/ds/z5UOcgxZzD?key=bwx9LQUT0t'

train_ann = 'train/_annotations.coco.json'
train_data = 'train/'
val_ann = 'valid/_annotations.coco.json'
val_data = 'valid/'

height = 300
width = 300
imgsz = (width, height)
batch = 16
workers = 4
val_batch = batch
val_workers = workers

# TRAIN
lr = 0.001
epochs = 300

weight_decay = 0.0005
momentum = 0.9

# ================================END=================================


model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor', mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True, pad_size_divisor=1
    ),
    backbone=dict(
        type='SSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://vgg16_caffe'),
    ),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20,
    ),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=num_classes,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=height,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        ),
        bbox_coder=dict(type='DeltaXYWHBBoxCoder', target_means=[0.0, 0.0, 0.0, 0.0], target_stds=[0.1, 0.1, 0.2, 0.2]),
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.0,
            ignore_iof_thr=-1,
            gt_max_assign_all=False,
        ),
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=1.0,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=1000, nms=dict(type='nms', iou_threshold=0.45), min_bbox_size=0, score_thr=0.02, max_per_img=200
    ),
)
cudnn_benchmark = True

backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=[0.0, 0.0, 0.0], to_rgb=True, ratio_range=(1, 4)),
    dict(type='MinIoURandomCrop', min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3),
    dict(type='Resize', scale=imgsz, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=imgsz, keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    batch_sampler=None,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img=train_data),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img=val_data),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + val_ann,
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator


# optimizer
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='SGD', lr=lr, momentum=momentum, weight_decay=weight_decay)
)

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW'),
]

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=batch)
