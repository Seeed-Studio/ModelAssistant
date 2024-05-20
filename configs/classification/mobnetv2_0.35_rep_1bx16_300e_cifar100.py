# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = '../_base_/default_runtime_cls.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# MODEL
widen_factor = 0.35

# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(type='MobileNetv2', widen_factor=widen_factor, out_indices=(2,), rep=True),
    neck=dict(type='sscma.GlobalAveragePooling'),
    head=dict(
        type='sscma.LinearClsHead',
        in_channels=16,
        loss=dict(type='sscma.CrossEntropyLoss', loss_weight=1.0),
    ),
)
