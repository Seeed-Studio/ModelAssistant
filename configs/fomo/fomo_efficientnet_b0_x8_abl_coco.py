_base_ = './fomo_mobnetv2_0.35_x8_abl_coco.py'

num_classes=2
model = dict(
    type='Fomo',
    backbone=dict(type='EfficientNet',arch='b0', out_indices=(3, )),
    # backbone=dict(type='edgeEfficienNet',arch='b0', out_indices=(2, )),
    head=dict(
        type='FomoHead',
        input_channels=40,
        num_classes=num_classes,
        middle_channels=96,
        act_cfg='ReLU6',
        loss_cls=dict(type='BCEWithLogitsLoss',
                      reduction='none',
                      pos_weight=100),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
    ),
)