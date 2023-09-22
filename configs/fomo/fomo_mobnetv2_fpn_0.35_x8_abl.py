_base_ = './fomo_mobnetv2_0.35_x8_abl_coco.py'

# ========================Suggested optional parameters========================
# MODEL
num_classes = 2
widen_factor = 0.35
# ================================END=================================

model = dict(
    backbone=dict(type='mmdet.MobileNetV2', widen_factor=widen_factor, out_indices=(2, 3, 5), _delete_=True),
    neck=dict(
        type='FPN',
        in_channels=[16, 24, 56],
        num_outs=3,
        out_idx=[
            0,
        ],
        out_channels=24,
    ),
    head=dict(type='FomoHead', input_channels=[24], num_classes=num_classes, act_cfg='ReLU'),
)
