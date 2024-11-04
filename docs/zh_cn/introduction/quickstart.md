# 快速上手

本节将指引如何快速使用SSCMA

以coco目标检测任务为例，整个流程如下：

- 环境安装
- 数据集准备
- 了解配置文件
- 模型训练
- 模型导出
- 推理测试


## 环境安装

环境安装我们强烈建议适应conda环境管理工具，对于conda的安装可查看Anaconda[官网](https://www.anaconda.com/)。

1. 首先执行以下命令来创建环境
```bash
# 创建一个名为sscma的虚拟环境，其中python版本为3.10
conda create -n sscma python=3.10 -y

# 激活创建的环境
conda activate sscma
```

2. 安装依赖库

```bash
# 获取仓库源码
git clone https://github.com/Seeed-Studio/SSCMA

# 安装pytorch 以下GPU和CPU二选一，根据自己的设备安装其中一个即可
# 安装GPU版本
conda install pytorch torchvision -c pytorch

#安装CPU版本
conda install pytorch torchvision cpuonly -c pytorch

# 安装其他依赖库
cd SSCMA
pip install -r requirements.txt 

```

## 数据集准备

数据集的格式默认使用COCO格式，其文件结构如下

```bash

-/datasets
    |
    |--/train
    |     |--train.json   #json文件为标注注释文件
    |     |--xxxx.jpg
    |     |--xxxx.jpg
    |     ...
    |--/val
          |--val.json    #json文件为标注注释文件
          |--xxxx.jpg
          |--xxxx.jpg
          ...
```


## 了解配置文件

1. 每个模型都有对应的配置文件,配置文件中确定了训练过程每个参数，对于新手可了解配置文件中的一些基本设置以及如何修改这些配置

  - 如下是rtmdet模型的一些基本配置，可能需要修改的为batch size及图片大小

```python
from mmengine.config import read_base

with read_base():
    from ._base_.default_runtime import *
    from .schedules.schedule_1x import *
    from .datasets.coco_detection import *

from torchvision.ops import nms
from torch.nn import SiLU, ReLU6, SyncBatchNorm,ReLU
from torch.optim.adamw import AdamW

from mmengine.hooks import EMAHook
from mmengine.optim import OptimWrapper, CosineAnnealingLR, LinearLR, AmpOptimWrapper
from sscma.datasets.transforms import (Resize,
    MixUp,
    Mosaic,
    Pad,
    RandomCrop,
    RandomFlip,
    Resize,
    HSVRandomAug,
    toTensor,
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
from sscma.deploy.models import RTMDetInfer


default_hooks.visualization = dict(
    type=DetVisualizationHook, draw=True, test_out_dir="works"
)

visualizer = dict(type=DetLocalVisualizer, vis_backends=vis_backends, name="visualizer")
# 模型深度与宽度因子
d_factor = 1
w_factor = 1
# 数据集类别数
num_classes = 80
# 模型输入图片大小
imgsz = (640, 640)
# 训练总轮数
max_epochs = 300
# 无Maisco的训练轮数
stage2_num_epochs = 20
# 学习率
base_lr = 0.004
# 训练时验证模型的间隔轮数
interval = 5
# batch大小
batch_size = 64
# 加载数据进程数
num_workers = 8


# 模型配置
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
        channel_attention=False,
        norm_cfg=dict(type=SyncBatchNorm),
        act_cfg=dict(type=ReLU, inplace=True),
    ),
    neck=dict(
        type=CSPNeXtPAFPN,
        deepen_factor=d_factor,
        widen_factor=w_factor,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type=SyncBatchNorm),
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
            norm_cfg=dict(type=SyncBatchNorm),
            act_cfg=dict(type=ReLU, inplace=True),
            share_conv=True,
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

# 推理测试时的推理配置
deploy = dict(
    type=RTMDetInfer,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[0, 0, 0],
        std=[255, 255, 255],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
)

# 训练阶段的数据增强相关配置
train_pipeline = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend="pillow",
        backend_args=None,
    ),
    dict(type=LoadAnnotations, imdecode_backend="pillow", with_bbox=True),
    dict(type=HSVRandomAug),
    dict(type=toTensor),
    dict(type=Mosaic, img_scale=imgsz, pad_val=114.0),
    # dict(type=Resize,scale=imgsz, keep_ratio=True),
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
        img_scale=imgsz,
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=114.0,
    ),
    dict(type=PackDetInputs),
]

# 在无mosaic阶段的数据增强配置
train_pipeline_stage2 = [
    dict(
        type=LoadImageFromFile,
        imdecode_backend="pillow",
        backend_args=None,
    ),
    dict(type=LoadAnnotations, imdecode_backend="pillow", with_bbox=True),
    dict(type=HSVRandomAug),
    dict(type=toTensor),
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

# 验证阶段的数据增强配置
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=toTensor),
    dict(type=Resize, scale=imgsz, keep_ratio=True),
    dict(type=Pad, size=imgsz, pad_val=dict(img=(114, 114, 114))),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# 训练dataloader相关配置
train_dataloader.update(
    dict(
        batch_size=batch_size,
        num_workers=num_workers,
        batch_sampler=None,
        pin_memory=True,
        collate_fn=coco_collate,
        dataset=dict(pipeline=train_pipeline),
    )
)
# 验证测试dataloader相关配置
val_dataloader.update(
    dict(batch_size=32, num_workers=8, dataset=dict(pipeline=test_pipeline))
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

# 优化器相关配置
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)

# 学习率策略相关配置
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

# 训练过程中hook配置
default_hooks.update(
    dict(
        checkpoint=dict(
            interval=interval,
            max_keep_ckpts=3,  # only keep latest 3 checkpoints
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

```

- 数据集相关配置，主要需要修改的配置为数据集的根目录以及注释文件的相关路径

```python
from mmengine.dataset.sampler import DefaultSampler
from sscma.datasets import CocoDataset, coco_collate, BatchShapePolicy
from sscma.datasets.transforms import (
    LoadAnnotations,
    PackDetInputs,
    RandomFlip,
    Resize,
    LoadImageFromFile,
)
from sscma.evaluation import CocoMetric

# dataset settings
dataset_type = CocoDataset
# 数据集文件夹路径
data_root = "datasets/coco/"
# 训练图像大小
imgsz=(640,640)

backend_args = None

# Config of batch shapes. Only on val.
batch_shapes_cfg = dict(
    type=BatchShapePolicy,
    batch_size=32,
    img_size=imgsz[0],
    size_divisor=32,
    extra_pad_ratio=0.5,
)

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=PackDetInputs),
]
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    collate_fn=dict(type=coco_collate),
    sampler=dict(type=DefaultSampler, shuffle=True),
    # batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_train2017.json", # 训练集的标注注释文件路径，可根据自己数据集的文件名及路径
        data_prefix=dict(img="train2017/"),         # 训练集图片路径，相对data_root的路径
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val2017.json", # 验证集的标注注释文件路径，可根据自己数据集的文件名及路径
        data_prefix=dict(img="val2017/"),   # 验证集图片路径，相对data_root的路径
        test_mode=True,
        pipeline=test_pipeline,
        # batch_shapes_cfg=batch_shapes_cfg,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + "annotations/instances_val2017.json", # 验证集的标注注释文件路径，可根据自己数据集的文件名及路径修改
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator
```


# 模型训练

- 执行以下命令即可开始训练
  
```bash
python tools/train.py configs/rtmdet_nano_8xb32_300e_coco.py --cfg-options data_root=${你的数据集路径} 
```
::: tip
对于配置文件的修改可在训练的命令行中使用--cfg-options参数进行，例如修改imgs和训练集的图片文件夹的路径 --cfg-options imgsz=416,416 train_dataloader.dataset.data_prefix.img=val
:::

- 在训练完成后训练期间所有的数据会保存在work_dirs/rtmdet_nano_8xb32_300e_coco文件夹下

# 模型导出

- 在训练完成后一般需要将模型部署到对应的设备上，此时的模型一般是onnx格式或者部署框架对应的格式，sscma支持将模型转为onnx，tflite，hailo，save model格式。

- 使用以下命令将模型导出到onnx格式
```bash
python tools/export.py configs/rtmdet_nano_8xb32_300e_coco.py  work_dirs/rtmdet_nano_8xb32_300e_coco/epoch_300.pth --format onnx
```
导出的onnx模型与pth权重在相同路径下

# 推理测试

- 在将模型导出后通常需要验证导出模型的精度是否和pth精度对齐，例如使用了量化后其精度会有些许丢失，这时需要测试丢失精度达到了多少，能否满足我们的部署需求。
- 使用以下命令即可测试不同格式的模型文件

```bash
python tools/test.py configs/configs/rtmdet_nano_8xb32_300e_coco.py  work_dirs/rtmdet_nano_8xb32_300e_coco/epoch_300.onnx
```






