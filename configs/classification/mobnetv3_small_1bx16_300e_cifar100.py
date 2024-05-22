# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './mobnetv2_1.0_1bx16_300e_cifar100.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='sscma.MobileNetV3', arch='small', _delete_=True),
    neck=dict(type='sscma.GlobalAveragePooling'),
    head=dict(
        type='sscma.StackedLinearClsHead',
        in_channels=576,
        mid_channels=[1280],
        dropout_rate=0.2,
        act_cfg=dict(type='mmcls.HSwish'),
        loss=dict(type='sscma.CrossEntropyLoss', loss_weight=1.0),
        init_cfg=dict(type='mmcls.Normal', layer='Linear', mean=0.0, std=0.01, bias=0.0),
    ),
)
