from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *
    from ._base_.schedules.schedule_1x import *
    from .datasets.coco_detection import *

from torchvision.ops import nms
from torch.nn import ReLU, BatchNorm2d
from torch.optim.adamw import AdamW

from mmengine.hooks import EMAHook
from mmengine.optim import OptimWrapper, CosineAnnealingLR, LinearLR
from sscma.datasets.transforms import (
    MixUp,
    Mosaic,
    Pad,
    RandomCrop,
    RandomFlip,
    Resize,
    HSVRandomAug,
    RandomResize,
    LoadImageFromFile,
    LoadAnnotations,
    PackDetInputs,
)
from sscma.datasets import DetDataPreprocessor
from sscma.engine import PipelineSwitchHook, DetVisualizationHook
from sscma.models import (
    BboxOverlaps2D,
    MlvlPointGenerator,
    DistancePointBBoxCoder,
    BatchDynamicSoftLabelAssigner,
    CSPNeXtPAFPN,
    GIoULoss,
    QualityFocalLoss,
    ExpMomentumEMA,
    RTMDet,
    RTMDetHead,
    RTMDetSepBNHeadModule,
    CSPNeXt,
)
from sscma.visualization import DetLocalVisualizer


default_hooks.visualization = dict(
    type=DetVisualizationHook, draw=True, test_out_dir="works"
)

visualizer = dict(type=DetLocalVisualizer, vis_backends=vis_backends, name="visualizer")

d_factor = 0.33
w_factor = 0.25
num_classes = 80
imgsz = (640, 640)
max_epochs = 300
stage2_num_epochs = 20
base_lr = 0.0005
interval = 10
batch_size = 32
num_workers = 16

# ratio range for random resize
random_resize_ratio_range = (0.5, 2.0)
# Number of cached images in mosaic
mosaic_max_cached_images = 20
# Number of cached images in mixup
mixup_max_cached_images = 10

checkpoint = "/home/baozhu/storage/epoch_67.pth"
model = dict(
    type=RTMDet,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=True,
        batch_augments=None,
    ),
    backbone=dict(
        type=CSPNeXt,
        arch="P5",
        expand_ratio=0.5,
        deepen_factor=d_factor,
        widen_factor=w_factor,
        channel_attention=True,
        split_max_pool_kernel=True,
        norm_cfg=dict(type=BatchNorm2d),
        act_cfg=dict(type=ReLU, inplace=True),
        init_cfg=dict(type="Pretrained", prefix="backbone.", checkpoint=checkpoint),
    ),
    neck=dict(
        type=CSPNeXtPAFPN,
        deepen_factor=d_factor,
        widen_factor=w_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=3,
        expand_ratio=0.5,
        norm_cfg=dict(type=BatchNorm2d),
        act_cfg=dict(type=ReLU, inplace=True),
    ),
    bbox_head=dict(
        type=RTMDetHead,
        head_module=dict(
            type=RTMDetSepBNHeadModule,
            num_classes=80,
            in_channels=256,
            stacked_convs=2,
            feat_channels=256,
            widen_factor=w_factor,
            norm_cfg=dict(type=BatchNorm2d),
            act_cfg=dict(type=ReLU, inplace=True),
            share_conv=False,
            pred_kernel_size=1,
            featmap_strides=[8, 16, 32],
        ),
        prior_generator=dict(type=MlvlPointGenerator, offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type=DistancePointBBoxCoder),
        loss_cls=dict(
            type=QualityFocalLoss, use_sigmoid=True, beta=2.0, loss_weight=1.0
        ),
        loss_bbox=dict(type=GIoULoss, loss_weight=2.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type=BatchDynamicSoftLabelAssigner,
            num_classes=num_classes,
            topk=13,
            iou_calculator=dict(type=BboxOverlaps2D),
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type=nms, iou_threshold=0.65),
        max_per_img=300,
    ),
)

train_pipeline = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend="pillow",
        backend_args=None,
    ),
    dict(type=LoadAnnotations, imdecode_backend="pillow", with_bbox=True),
    dict(type=HSVRandomAug),
    dict(type=Mosaic,
         img_scale=imgsz,
         use_cached=True,
         max_cached_images=mosaic_max_cached_images,  # note
         random_pop=False,  # note
         pad_val=114.0
         ),
    dict(
        type=RandomResize,
        scale=(imgsz[0] * 2, imgsz[1] * 2),
        ratio_range=(0.1, 2.0),
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(
        type=MixUp,
        use_cached=True,
        random_pop=False,
        max_cached_images=mixup_max_cached_images,
        prob=0.5
    ),
    dict(type=PackDetInputs),
]

train_pipeline_stage2 = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend="pillow",
        backend_args=None,
    ),
    dict(type=LoadAnnotations, imdecode_backend="pillow", with_bbox=True),
    dict(type=HSVRandomAug),
    dict(
        type=RandomResize,
        scale=(imgsz[0] * 2, imgsz[1] * 2),
        ratio_range=(0.1, 2.0),
        resize_type=Resize,
        keep_ratio=True,
    ),
    dict(type=RandomCrop, crop_size=imgsz),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(type=PackDetInputs),
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=imgsz, keep_ratio=True),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

train_dataloader.update(
    dict(
        batch_size=batch_size,
        num_workers=num_workers,
        batch_sampler=None,
        pin_memory=True,
        collate_fn=coco_collate,
        dataset=dict(pipeline=train_pipeline,
                     ann_file="annotations/instances_minitrain2017.json"),
    )
)

val_dataloader.update(
    dict(batch_size=16, num_workers=8, dataset=dict(pipeline=test_pipeline))
)
test_dataloader = val_dataloader


train_cfg.update(
    dict(
        max_epochs=max_epochs,
        val_interval=interval,
        dynamic_intervals=[(max_epochs - stage2_num_epochs, 1)],
    )
)

val_evaluator.update(dict(proposal_nums=(100, 1, 10)))
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# learning rate
param_scheduler = [
    dict(type=LinearLR, start_factor=1.0e-5, by_epoch=False, begin=0, end=1000),
    dict(
        # use cosine lr from 150 to 300 epoch
        type=CosineAnnealingLR,
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

# hooks
default_hooks.update(
    dict(
        checkpoint=dict(
            interval=interval,
            max_keep_ckpts=3,  # only keep latest 3 checkpoints
            save_best='auto'
        )
    )
)

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type=ExpMomentumEMA,
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
    dict(
        type=PipelineSwitchHook,
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
    ),
]

dump_config = True
