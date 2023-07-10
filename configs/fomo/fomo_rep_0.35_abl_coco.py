_base_ = './fomo_mobnetv2_0.35_x8_abl_coco.py'

num_classes = 2
height = 192
width = 192
img_scale = (width, height)  # width, height

affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio
loss_cls_weight = 0.5
loss_bbox_weight = 0.05
loss_obj_weight = 1.0
prior_match_thr = 4.0  # Priori box matching threshold

strides = [8, 16, 32]
anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)],  # P5/32
]
num_det_layers = 3  # The number of model output scales
norm_cfg = dict(type='BN', momentum=0.03, eps=0.001)  # Normalization config
obj_level_weights = [4.0, 1.0, 0.4]
model = dict(
    type='Fomo',
    backbone=dict(type='MobileNetv2', widen_factor=0.35, out_indices=(2, 3, 4), rep=True),
    head=dict(
        type='edgelab.YOLOV5Head',
        head_module=dict(
            type='edgelab.DetHead',
            num_classes=num_classes,
            in_channels=[16, 32, 64],
            widen_factor=1,
            featmap_strides=strides,
            num_base_priors=3,
        ),
        prior_generator=dict(type='mmdet.YOLOAnchorGenerator', base_sizes=anchors, strides=strides),
        # scaled based on number of detection layers
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_cls_weight * (num_classes / 80 * 3 / num_det_layers),
        ),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=loss_bbox_weight * (3 / num_det_layers),
            return_iou=True,
        ),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=loss_obj_weight * ((img_scale[0] / 640) ** 2 * 3 / num_det_layers),
        ),
        prior_match_thr=prior_match_thr,
        obj_level_weights=obj_level_weights,
    ),
)
