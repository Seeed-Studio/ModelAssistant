_base_ = ["./base_arch.py"]

anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]
num_classes = 11
deepen_factor = 0.33
widen_factor = 0.25

strides = [8, 16, 32]

model = dict(
    type='mmyolo.YOLODetector',
    data_preprocessor=dict(type='mmdet.DetDataPreprocessor',
                           mean=[0., 0., 0.],
                           std=[255., 255., 255.],
                           bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
    ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
        ),
    ),
)