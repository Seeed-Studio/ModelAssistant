_base_ = '../_base_/default_runtime_pose.py'

num_classes = 4
model = dict(
    type='PFLD',
    backbone=dict(
        type='PfldMobileNetV2',
        inchannel=3,
        layer1=[16, 16, 16, 16, 16],
        layer2=[32, 32, 32, 32, 32, 32],
        out_channel=32,
    ),
    head=dict(type='PFLDhead', num_point=num_classes, input_channel=32, act_cfg="ReLU", loss_cfg=dict(type='PFLDLoss')),
)

# dataset settings
dataset_type = 'MeterData'

data_root = ""
height = 112
width = 112
batch_size = 32
workers = 4

train_pipeline = [
    dict(type="Resize", height=height, width=width, interpolation=0),
    # dict(type="PixelDropout"),
    dict(type='ColorJitter', brightness=0.3, contrast=0.3, saturation=0.3, p=0.5),
    # dict(type="CoarseDropout",max_height=12,max_width=12),
    # dict(type='GaussNoise'),
    dict(type='MedianBlur', blur_limit=3, p=0.5),
    dict(type='HorizontalFlip'),
    dict(type='VerticalFlip'),
    dict(type='Rotate', limit=45, p=0.7),
    dict(type='Affine', translate_percent=[0.05, 0.30], p=0.6),
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

lr = 0.0001
epochs = 1000
evaluation = dict(save_best='loss')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=lr, betas=(0.9, 0.99), weight_decay=1e-6))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
val_evaluator = dict(type='PointMetric')
test_evaluator = val_evaluator

find_unused_parameters = True

train_cfg = dict(by_epoch=True, max_epochs=epochs)
# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001, by_epoch=False),  # warm-up
    dict(type='MultiStepLR', begin=1, end=500, milestones=[350, 500, 600, 700, 800, 900], gamma=0.1, by_epoch=True),
]
