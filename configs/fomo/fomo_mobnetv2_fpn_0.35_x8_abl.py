_base_ = './fomo_mobnetv2_0.35_x8_abl_coco.py'

num_classes = 2
model = dict(
    backbone=dict(type='mmdet.MobileNetV2',
                  widen_factor=0.35,
                  out_indices=(2, 3, 5)),
    neck=dict(
        type='mmdet.FPN',
        in_channels=[16, 24, 56],
        out_channels=24,
        num_outs=3,
    ),
    head=dict(
        type='FomoHead',
        input_channels=[24,24,24],
        out_channels=[2,2,2],
        num_classes=num_classes,
        middle_channels=[96, 32],
        act_cfg='ReLU6',
        loss_cls=dict(type='BCEWithLogitsLoss',
                      reduction='none',
                      pos_weight=40),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
        cls_weight=40,
    ),
)