_base_ = './pfld_mbv2n_112.py'

num_classes = 4
model = dict(
    type='PFLD',
    backbone=dict(
        type='MobileNetV3',
        inchannel=3,
        arch='large',
        out_indices=(3,),
    ),
    head=dict(type='PFLDhead', num_point=num_classes, input_channel=40, act_cfg="ReLU", loss_cfg=dict(type='PFLDLoss')),
)

# dataset settings
dataset_type = 'MeterData'

data_root = ""
height = 192
width = 192
batch_size = 32
workers = 4

train_pipeline = [
    dict(type="Resize", height=height, width=width, interpolation=0),
    # dict(type="PixelDropout"),
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
    # dict(type='GaussNoise'),
    # dict(type="CoarseDropout",max_height=12,max_width=12),
    dict(type='MedianBlur', blur_limit=3, p=0.5),
    dict(type='HorizontalFlip'),
    dict(type='VerticalFlip'),
    dict(type='Rotate', limit=45, p=0.7),
    dict(type='Affine', translate_percent=[0.05, 0.3], p=0.6),
]

val_pipeline = [dict(type="Resize", height=height, width=width)]

train_dataloader = dict(
    batch_size=16,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="train/images",
        index_file=r'train/annotations.txt',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="val/images",
        index_file=r'val/annotations.txt',
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader
