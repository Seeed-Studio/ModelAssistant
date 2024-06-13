# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
_base_ = './fastestdet_shuffv2_spp_voc.py'

# ========================Suggested optional parameters========================
# MODEL
num_classes = (20,)
widen_factor=0.25,

# ================================END=================================

model = dict(
    type='FastestDet',
    backbone=dict(
        type='ShuffleNetV2',
        out_indices=(0, 1, 2),
        widen_factor=widen_factor,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
    ),
    neck=dict(type='SPP', input_channels=336, output_channels=96, layers=[1, 2, 3]),
    bbox_head=dict(
        type='Fomo_Head',
        input_channels=96,
        num_classes=num_classes,
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='mean'),
        loss_cls=dict(type='BCEWithLogitsLoss', reduction='mean'),
    ),
)

evaluation = dict(interval=1, metric=['mAP'], fomo=True)
