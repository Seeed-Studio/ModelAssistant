# Copyright (c) Seeed Tech Ltd. All rights reserved.
_base_ = ['../_base_/default_runtime_det.py']
default_scope = 'sscma'

# ========================Suggested optional parameters========================
# DATA
# Dataset type, this will be used to define the dataset
# dataset_type = 'mmdet.CocoDataset'
supdataset_type = 'sscma.CustomYOLOv5CocoDataset'
unsupdataset_type = 'sscma.UnsupDataset'
# unsupdataset_type='sscma.CustomYOLOv5CocoDataset'
# Path to the dataset's root directory
# dataset link: https://universe.roboflow.com/team-roboflow/coco-128
data_root = 'https://universe.roboflow.com/ds/z5UOcgxZzD?key=bwx9LQUT0t'

# Path of train annotation file
# train_ann = 'train/_annotations.coco.json'
train_ann = 'annotations/train_sup1.json'
# Prefix of train image path
train_data = 'train2017/'
# Path of val annotation file
# val_ann = 'valid/_annotations.coco.json'
val_ann = 'annotations/instances_val2017.json'
# Prefix of val image path
val_data = 'val2017/'
# Height of the model input data
height = 640
# Width of the model input data
width = 640
# The width and height of the model input data
imgsz = (width, height)  # width, height

# MODEL
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.33
# The scaling factor that controls the width of the network structure
widen_factor = 0.5
# Number of classes for classification
num_classes = 80

# TRAIN
# Base learning rate for optim_wrapper. Corresponding to 8xb16=128 bs
lr = 0.01
# Total number of rounds of model training
epochs = 500
# Number of input data per iteration in the model training phase
batch = 32
# Number of threads used to load data during training, this value should be adjusted accordingly to the training batch
workers = 1
# Optimizer weight decay value
weight_decay = 0.0005
# SGD momentum/Adam beta1
momentum = 0.937
# Learning rate scaling factor
lr_factor = 0.01
# persistent_workers must be False if num_workers is 0
persistent_workers = True

# VAL
# Batch size of a single GPU during validation
val_batch = 16
# Worker to pre-fetch data for each single GPU during validation
val_workers = 2
# Save model checkpoint and validation intervals
val_interval = 1
# Model weight saving interval in epochs
save_interval = val_interval
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# ================================END=================================

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.1,  # Threshold to filter out boxes.
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300,
)  # Max number of detections of each image


# Strides of multi-scale prior box
strides = [8, 16, 32]
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config

anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)],  # P5/32
]

# -----train val related-----
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
prior_match_thr = 4.0  # Priori box matching threshold
# The obj loss weights of the three output layers
obj_level_weights = [4.0, 1.0, 0.4]

# Single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)

# model arch
detector = dict(
    type='sscma.YOLODetector',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor', mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True
    ),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    bbox_head=dict(
        type='sscma.YOLOV5Head',
        head_module=dict(
            type='sscma.DetHead',
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
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
    # _delete_=True
)
model = dict(
    type='sscma.BaseSsod',
    detector=detector,
    pseudo_label_cfg=dict(
        type='sscma.LabelMatch',
        # cfg=dict(
        #     multi_label=False,
        #     conf_thres=0.1,
        #     iou_thres=0.65,
        #     ignore_thres_high=0.6,
        #     ignore_thres_low=0.1,
        #     resample_high_percent=0.25,
        #     resample_low_percent=0.99,
        #     data_names=('person',),
        #     data_np=0,
        # ),
        # target_data_len=10,
        # label_num_per_img=10,
        # nc=80,
    ),
    teacher_loss_weight=0,
    # da_loss_weight=0,
    data_preprocessor=dict(type='mmdet.MultiBranchDataPreprocessor', data_preprocessor=detector['data_preprocessor']),
    # data_preprocessor=dict(
    # type='mmdet.DetDataPreprocessor', mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True
    # ),
    semi_train_cfg=dict(
        freeze_teacher=True,
        sup_weight=1.0,
        unsup_weight=4.0,
        pseudo_label_initial_score_thr=0.5,
        rpn_pseudo_thr=0.9,
        cls_pseudo_thr=0.9,
        reg_pseudo_thr=0.02,
        jitter_times=10,
        jitter_scale=0.06,
    ),
    semi_test_cfg=dict(predict_on='teacher'),
    # _delete_=True
)

color_space = [
    [dict(type='mmdet.ColorTransform')],
    [dict(type='mmdet.AutoContrast')],
    [dict(type='mmdet.Equalize')],
    [dict(type='mmdet.Sharpness')],
    [dict(type='mmdet.Posterize')],
    [dict(type='mmdet.Solarize')],
    [dict(type='mmdet.Color')],
    [dict(type='mmdet.Contrast')],
    [dict(type='mmdet.Brightness')],
]

# geometric = [
#     [dict(type='mmdet.Rotate')],
#     [dict(type='mmdet.ShearX')],
#     [dict(type='mmdet.ShearY')],
#     [dict(type='mmdet.TranslateX')],
#     [dict(type='mmdet.TranslateY')],
# ]
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
]


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

# pipeline used to augment unlabeled data weakly,
# which will be sent to teacher model for predicting pseudo instances.
weak_pipeline = [
    # dict(type='Mosaic', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(type='LetterResize', scale=imgsz, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'homography_matrix',
        ),
    ),
]
# pipeline used to augment unlabeled data strongly,
# which will be sent to student model for unsupervised training.
strong_pipeline = [
    # dict(type='Mosaic', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(type='LetterResize', scale=imgsz, allow_scale_up=False, pad_val=dict(img=114)),
    # dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.RandomOrder',
        transforms=[
            dict(type='mmdet.RandAugment', aug_space=color_space, aug_num=1),
            # dict(type='mmdet.RandAugment', aug_space=geometric, aug_num=1),
        ],
    ),
    dict(type='mmdet.RandomErasing', n_patches=(1, 5), ratio=(0, 0.2)),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            # 'flip',
            # 'flip_direction',
            'homography_matrix',
        ),
    ),
]
sup_branch_field = ['sup', 'unsup_teacher', 'unsup_student']
unsup_branch_field = ['sup', 'unsup_teacher', 'unsup_student']
# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    *pre_transform,
    dict(type='Mosaic', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(type='LetterResize', scale=imgsz, allow_scale_up=False, pad_val=dict(img=114)),
    # dict(type='mmdet.RandomResize', scale=imgsz, keep_ratio=True),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(type='mmdet.RandAugment', aug_space=color_space, aug_num=1),
    dict(type='mmdet.FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='mmdet.MultiBranch', branch_field=sup_branch_field, sup=dict(type='mmdet.PackDetInputs')),
]
# pipeline used to augment unlabeled data into different views
unsup_pipeline = [
    *pre_transform,
    dict(
        type='mmdet.MultiBranch',
        branch_field=unsup_branch_field,
        unsup_teacher=weak_pipeline,
        unsup_student=strong_pipeline,
    ),
]
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01),
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
        type='mmdet.MultiBranch',
        branch_field=sup_branch_field,
        sup=dict(
            type='mmdet.PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction'),
        ),
    ),
]

labeled_dataset = dict(
    type=supdataset_type,
    data_root=data_root,
    ann_file=train_ann,
    data_prefix=dict(img=train_data),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline,
)

unlabeled_dataset = dict(
    type=unsupdataset_type,
    data_root=data_root,
    ann_file=val_ann,
    data_prefix=dict(img='unlabel_data'),
    filter_cfg=dict(filter_empty_gt=False),
    pipeline=unsup_pipeline,
)


train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=True,
    sampler=dict(
        # type='mmdet.GroupMultiSourceSampler',
        type='sscma.SemiSampler',
        batch_size=batch,
        sample_ratio=[1, 4],
        round_up=True,
    ),
    dataset=dict(type='sscma.SemiDataset', sup_dataset=labeled_dataset, unsup_dataset=unlabeled_dataset),
)

val_dataloader = dict(
    batch_size=val_batch,
    num_workers=val_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=supdataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data),
        ann_file=val_ann,
        pipeline=test_pipeline,
        # batch_shapes_cfg=None,
    ),
)

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True, batch_size_per_gpu=batch
    ),
    constructor='YOLOv5OptimizerConstructor',
)

default_hooks = dict(
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook', scheduler_type='linear', lr_factor=lr_factor, max_epochs=epochs
    ),
    checkpoint=dict(type='CheckpointHook', interval=val_interval, save_best='auto', max_keep_ckpts=max_keep_ckpts),
)


custom_hooks = [
    dict(
        type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, strict_load=False, priority=49
    ),
    dict(type='mmdet.MeanTeacherHook'),
    dict(type='sscma.SemiHook', bure_epoch=200),
    dict(type='sscma.LabelMatchHook', priority=100),
]


val_evaluator = dict(type='mmdet.CocoMetric', proposal_nums=(100, 1, 10), ann_file=data_root + val_ann, metric='bbox')
test_evaluator = val_evaluator

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epochs, val_interval=val_interval, _delete_=True)
val_cfg = dict(type='sscma.SemiValLoop', bure_epoch=200)
test_cfg = val_cfg
