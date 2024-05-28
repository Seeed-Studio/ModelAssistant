# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './mobnetv2_1.0_1bx16_300e_cifar10.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# MODEL
gray = False
widen_factor = 0.35

# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='MobileNetV2', gray_input=gray, widen_factor=widen_factor, out_indices=(2,), rep=True),
    neck=dict(type='sscma.GlobalAveragePooling'),
    head=dict(
        type='sscma.LinearClsHead',
        in_channels=32,
        loss=dict(type='sscma.CrossEntropyLoss', loss_weight=1.0),
    ),
)
