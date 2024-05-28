# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './mobnetv2_1.0_1bx16_300e_custom.py'
default_scope = 'sscma'
custom_imports = dict(imports=['sscma'], allow_failed_imports=False)

# ========================Suggested optional parameters========================
# MODEL
gray = False

# ================================END=================================

model = dict(
    type='sscma.ImageClassifier',
    backbone=dict(
        type='EfficientNet', arch='b0', input_channels=1 if gray else 3, out_indices=(6,), rep=True, _delete_=True
    ),
    neck=dict(type='sscma.GlobalAveragePooling'),
    head=dict(
        type='sscma.LinearClsHead',
        in_channels=320,
        loss=dict(type='sscma.CrossEntropyLoss', loss_weight=1.0),
    ),
)
