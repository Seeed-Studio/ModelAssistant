# Copyright (c) Seeed Tech Ltd. All rights reserved.
_base_ = './mobnetv2_1.0_1bx16_300e_cifar10.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# MODEL
widen_factor = 1.0

# ================================END=================================
model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='MobileNetv2', widen_factor=widen_factor, rep=True),
    neck=dict(type='mmcls.GlobalAveragePooling'),
    head=dict(
        type='mmcls.LinearClsHead',
        in_channels=128,
    ),
)
