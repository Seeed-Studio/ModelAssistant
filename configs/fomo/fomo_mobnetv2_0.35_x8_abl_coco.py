_base_ = "../_base_/default_runtime_det.py"
default_scope = "edgelab"
custom_imports = dict(imports=["edgelab"], allow_failed_imports=False)

num_classes = 2
data_preprocessor = dict(type='mmdet.DetDataPreprocessor',
                         mean=[0, 0, 0],
                         std=[255., 255., 255.],
                         bgr_to_rgb=True,
                         pad_size_divisor=32)
model = dict(
    type="Fomo",
    data_preprocessor=data_preprocessor,
    backbone=dict(type="MobileNetv2",
                  widen_factor=0.35,
                  out_indices=(2, ),
                  rep=True),
    head=dict(
        type="FomoHead",
        input_channels=[16],
        num_classes=num_classes,
        middle_channel=48,
        act_cfg="ReLU6",
        loss_cls=dict(type="BCEWithLogitsLoss",
                      reduction="none",
                      pos_weight=40),
        loss_bg=dict(type="BCEWithLogitsLoss", reduction="none"),
    ),
)

# dataset settings
dataset_type = "CustomCocoDataset"
data_root = ""
height = 96
width = 96
batch_size = 16
workers = 1

albu_train_transforms = [
    dict(type="RandomResizedCrop",
         height=height,
         width=width,
         scale=(0.80, 1.2),
         p=1),
    dict(type="Rotate", limit=30),
    dict(type="RandomBrightnessContrast",
         brightness_limit=0.3,
         contrast_limit=0.3,
         p=0.5),
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01),
    dict(type="HorizontalFlip", p=0.5),
]
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='mmdet.LoadAnnotations', with_bbox=True)
]
train_pipeline = [
    *pre_transform,
    dict(type='mmdet.Albu',
         transforms=albu_train_transforms,
         bbox_params=dict(type='BboxParams',
                          format='pascal_voc',
                          label_fields=['gt_bboxes_labels',
                                        'gt_ignore_flags']),
         keymap={
             'img': 'image',
             'gt_bboxes': 'bboxes'
         }),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_path', 'img_id', 'instances', 'img_shape',
                    'ori_shape', 'gt_bboxes', 'gt_bboxes_labels'))
]

test_pipeline = [
    *pre_transform,
    dict(type="mmdet.Resize", scale=(height, width)),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor'))
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="train/_annotations.coco.json",
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="valid/_annotations.coco.json",
        data_prefix=dict(img='valid/'),
        pipeline=test_pipeline,
    ),
)
test_dataloader = val_dataloader

# data_preprocessor=dict(type='mmdet.DetDataPreprocessor')
# optimizer
lr = 0.001
epochs = 100

find_unused_parameters = True

optim_wrapper = dict(
    optimizer=dict(type="Adam", lr=lr, weight_decay=5e-4, eps=1e-7))

# evaluator
val_evaluator = dict(type="FomoMetric")
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs=epochs)

# learning policy
param_scheduler = [
    dict(type="LinearLR", begin=0, end=30, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(
        type="MultiStepLR",
        begin=1,
        end=500,
        milestones=[100, 200, 250],
        gamma=0.1,
        by_epoch=True,
    ),
]
