_base_ = './fomo_mobnetv2_0.35_x8_abl_coco.py'

num_classes=2
model = dict(
    type='Fomo',
    backbone=dict(type='ShuffleNetV2', widen_factor=0.1, out_indices=(1, )),
    head=dict(
        type='Fomo_Head',
        input_channels=24,
        num_classes=num_classes,
        middle_channels=[96, 32],
        act_cfg='ReLU6',
        loss_cls=dict(type='BCEWithLogitsLoss',
                      reduction='none',
                      pos_weight=100),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
        cls_weight=100,
    ),
)