_base_ = './base.py'

# ========================Suggested optional parameters========================
# MODEL
deepen_factor = 1.00
widen_factor = 1.00

# DATA
height = 640
width = 640
imgsz = (width, height)

# ================================END=================================
last_stage_out_channels = 512

mixup_prob = 0.15
affine_scale = 0.9

# =======================Unmodified in most cases==================

model = dict(
    backbone=dict(
        last_stage_out_channels=last_stage_out_channels, deepen_factor=deepen_factor, widen_factor=widen_factor
    ),
    neck=dict(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, last_stage_out_channels],
        out_channels=[256, 512, last_stage_out_channels],
    ),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor, in_channels=[256, 512, last_stage_out_channels])),
)

pre_transform = [dict(type='LoadImageFromFile', backend_args=None), dict(type='LoadAnnotations', with_bbox=True)]

mosaic_affine_transform = [
    dict(type='Mosaic', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_aspect_ratio=100,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        # imgsz is (width, height)
        border=(-imgsz[0] // 2, -imgsz[1] // 2),
        border_val=(114, 114, 114),
    ),
]
albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01),
]

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
    *mosaic_affine_transform,
    dict(type='YOLOv5MixUp', prob=mixup_prob, pre_transform=[*pre_transform, *mosaic_affine_transform]),
    *last_transform,
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
