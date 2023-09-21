_base_ = '../_base_/default_runtime_det.py'
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

height = 320
width = 320
imgsz = (width, height)


# TRAIN
lr = 0.001
epochs = 300
batch = 32
workers = 2
val_batch=1
val_workers=1

weight_decay = 0.0005
momentum = 0.9

# ================================END=================================
# model settings

data_preprocessor = dict(
    type='DetDataPreprocessor', mean=[0, 0, 0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True, pad_size_divisor=32
)
model = dict(
    type='YOLOV3',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        # init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')
    ),
    neck=dict(type='YOLOV3Neck', num_scales=3, in_channels=[1024, 512, 256], out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=num_classes,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[
                [(116, 90), (156, 198), (373, 326)],
                [(30, 61), (62, 45), (59, 119)],
                [(10, 13), (16, 30), (33, 23)],
            ],
            strides=[32, 16, 8],
        ),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, reduction='sum'),
        loss_conf=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0, reduction='sum'),
        loss_xy=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=2.0, reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum'),
    ),
    # training and testing settings
    train_cfg=dict(assigner=dict(type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100,
    ),
)


backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=data_preprocessor['mean'], to_rgb=data_preprocessor['bgr_to_rgb'], ratio_range=(1, 2)),
    dict(type='MinIoURandomCrop', min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), min_crop_size=0.3),
    dict(type='RandomResize', scale=[(320, 320), imgsz], keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=imgsz, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]


train_dataloader = dict(
    batch_size=batch,
    num_workers=workers,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
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

# evaluator
val_evaluator = dict(type='CocoMetric', ann_file=data_root + val_ann, metric='bbox', backend_args=backend_args)
test_evaluator = val_evaluator


find_unused_parameters = True

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=lr, momentum=momentum, weight_decay=weight_decay),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# learning policy
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=2000),
    dict(type='MultiStepLR', by_epoch=True, milestones=[218, 246], gamma=0.1),
]

train_cfg = dict(by_epoch=True, max_epochs=epochs)
val_cfg = dict()
test_cfg = dict()
# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=30, start_factor=0.001, by_epoch=False),  # warm-up
    dict(type='MultiStepLR', begin=1, end=500, milestones=[100, 200, 250], gamma=0.1, by_epoch=True),
]
