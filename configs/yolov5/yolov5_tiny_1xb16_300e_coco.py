_base_ = ['./base_arch.py']

anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)],  # P5/32
]

# ======================modify start======================

# model
strides = [8, 16, 32]
num_classes = 80
deepen_factor = 0.33
widen_factor = 0.15

# datasets
data_root = ''
height = 192
width = 192
batch_size = 16
workers = 2

# training
lr = 0.01
epochs = 300  # Maximum training epochs

# ======================modify end======================

# ======================model==================
model = dict(
    type='mmyolo.YOLODetector',
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
        ),
    ),
)

# ======================datasets==================
img_scale = (width, height)
affine_scale = 0.5
persistent_workers = True
dataset_type = 'edgelab.CustomYOLOv5CocoDataset'
train_ann_file = 'train/_annotations.coco.json'
train_data_prefix = 'train/'  # Prefix of train image path
val_ann_file = 'valid/_annotations.coco.json'
val_data_prefix = 'valid/'  # Prefix of val image path

batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=1,
    img_size=img_scale[0],
    # The image scale of padding should be divided by pad_size_divisor
    size_divisor=32,
    # Additional paddings for pixel scale
    extra_pad_ratio=0.5,
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
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
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
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann_file,
        data_prefix=dict(img=train_data_prefix),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    ),
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(type='LetterResize', scale=img_scale, allow_scale_up=False, pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param'),
    ),
]

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(img=val_data_prefix),
        ann_file=val_ann_file,
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg,
    ),
)

test_dataloader = val_dataloader

# ======================training==================
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=lr, momentum=0.937, weight_decay=0.0005, nesterov=True, batch_size_per_gpu=batch_size
    ),
    constructor='YOLOv5OptimizerConstructor',
)

val_evaluator = dict(
    type='mmdet.CocoMetric', proposal_nums=(100, 1, 10), ann_file=data_root + val_ann_file, metric='bbox'
)
test_evaluator = val_evaluator
