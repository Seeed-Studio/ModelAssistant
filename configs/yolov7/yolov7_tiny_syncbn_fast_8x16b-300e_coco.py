_base_ = './base.py'

# ========================Suggested optional parameters========================
# DATA
height = 640
width = 640
imgsz = (width, height)  # width, height
num_classes = 71

# TRAIN
lr_factor = 0.01  # Learning rate scaling factor
# ================================END=================================
# -----model related-----
# Data augmentation
max_translate_ratio = 0.1  # YOLOv5RandomAffine
scaling_ratio_range = (0.5, 1.6)  # YOLOv5RandomAffine
mixup_prob = 0.05  # YOLOv5MixUp
randchoice_mosaic_prob = [0.8, 0.2]
mixup_alpha = 8.0  # YOLOv5MixUp
mixup_beta = 8.0  # YOLOv5MixUp

# -----train val related-----
loss_cls_weight = 0.5
loss_obj_weight = 1.0
num_det_layers = 3

# ===============================Unmodified in most cases====================

model = dict(
    backbone=dict(arch='Tiny', act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    neck=dict(
        is_tiny_version=True,
        in_channels=[128, 256, 512],
        out_channels=[64, 128, 256],
        block_cfg=dict(_delete_=True, type='TinyDownSampleBlock', middle_ratio=0.25),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        use_repconv_outs=False,
    ),
    bbox_head=dict(
        head_module=dict(in_channels=[128, 256, 512]),
        loss_cls=dict(loss_weight=loss_cls_weight * (num_classes / 80 * 3 / num_det_layers)),
        loss_obj=dict(loss_weight=loss_obj_weight * ((imgsz[0] / 640) ** 2 * 3 / num_det_layers)),
    ),
)

pre_transform = [dict(type='LoadImageFromFile', backend_args=None), dict(type='LoadAnnotations', with_bbox=True)]

mosiac4_pipeline = [
    dict(type='Mosaic', img_scale=imgsz, pad_val=114.0, pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        max_translate_ratio=max_translate_ratio,  # change
        scaling_ratio_range=scaling_ratio_range,  # change
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
        max_translate_ratio=max_translate_ratio,  # change
        scaling_ratio_range=scaling_ratio_range,  # change
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
        alpha=mixup_alpha,
        beta=mixup_beta,
        prob=mixup_prob,  # change
        pre_transform=[*pre_transform, randchoice_mosaic_pipeline],
    ),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip', 'flip_direction')
    ),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
default_hooks = dict(param_scheduler=dict(lr_factor=lr_factor))
