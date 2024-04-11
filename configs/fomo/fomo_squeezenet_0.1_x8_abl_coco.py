# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './fomo_mobnetv2_0.35_x8_abl_coco.py'

# ========================Suggested optional parameters========================
# MODEL
num_classes = 2
widen_factor=0.1
# ================================END=================================
model = dict(
    type='Fomo',
    backbone=dict(type='SqueezeNet', widen_factor=widen_factor, out_indices=(2,), _delete_=True),
    head=dict(
        type='FomoHead',
        input_channels=24,
        num_classes=num_classes,
        middle_channel=[96, 32],
        act_cfg='ReLU6',
        loss_cls=dict(type='BCEWithLogitsLoss', reduction='none', pos_weight=100),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
    ),
)
