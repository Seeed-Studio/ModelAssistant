_base_ = '../_base_/default_runtime.py'

num_classes = 20
model = dict(
    type='FastestDet',
    backbone=dict(
        type='ShuffleNetV2',
        out_indices=(0, 1, 2),
        widen_factor=0.25,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    ),
    neck=dict(type='SPP', input_channels=336, output_channels=96, layers=[1, 2, 3]),
    bbox_head=dict(
        type='Fastest_Head',
        input_channels=96,
        num_classes=num_classes,
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

# dataset settings
dataset_type = 'CustomVocdataset'
data_root = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
height = 352
width = 352
batch_size = 32
workers = 4

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Expand', mean=img_norm_cfg['mean'], to_rgb=img_norm_cfg['to_rgb'], ratio_range=(1, 2)),
    dict(type='MinIoURandomCrop', min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), min_crop_size=0.3),
    dict(type='Resize', img_scale=[(height, width)], multiscale_mode='range', keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(height, height),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ],
    ),
]
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=workers,
    train=dict(
        type='RepeatDataset',  # use RepeatDataset to speed up training
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='ImageSets/Main/train.txt',
            #  img_prefix=None,
            pipeline=train_pipeline,
        ),
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/val.txt',
        #  img_prefix=None,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/val.txt',
        #   img_prefix=None,
        pipeline=test_pipeline,
    ),
)

# optimizer
lr = 0.001
max_epochs = 300
optimizer = dict(type='SGD', lr=lr, momentum=0.949, weight_decay=0.0005)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', warmup='linear', warmup_iters=8000, warmup_ratio=0.000001, step=[100, 200, 250])
# runtime settings
evaluation = dict(interval=1, metric=['mAP'])
find_unused_parameters = True

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (24 samples per GPU)
auto_scale_lr = dict(base_batch_size=192)
