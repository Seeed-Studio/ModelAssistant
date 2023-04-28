#  模型配置

EdgeLab 使用 [OpenMMLab - MMEngine](https://github.com/open-mmlab/mmengine) 提供的配置处理系统，具有模块化、可继承的设计特点，为用户提供了统一的配置访问接口，便于用户对不同的神经网络进行各种测试与验证。


## 配置的目录结构

EdgeLab 使用的配置文件位于 `configs` 目录下，用于不同任务下不同模型的训练。我们在其根据不同的任务分类划分了子文件夹，在各个子文件夹中，保存有多个模型的不同训练管线参数，配置文件的目录结构如下:

::: code-group

``` [整体结构]
configs
├── _base_
│   ├── datasets
│   │   └── coco_detection.py
│   ├── default_runtime_cls.py
│   ├── default_runtime_det.py
│   ├── default_runtime_pose.py
│   └── schedules
│       ├── schedule_1x.py
│       ├── schedule_20e.py
│       └── schedule_2x.py
├── fomo
│   ├── fomo_efficientnet_b0_x8_abl_coco.py
│   ├── fomo_mobnetv2_0.35_x8_abl_coco.py
│   ├── fomo_mobnetv2_fpn_0.35_x8_abl.py
│   ├── fomo_mobnetv2_x8_coco.py
│   ├── fomo_mobnetv2_x8_voc.py
│   ├── fomo_mobnetv3_0.35_x8_abl_coco.py
│   ├── fomo_shufflenetv2_0.1_x8_abl_coco.py
│   └── fomo_squeezenet_0.1_x8_abl_coco.py
├── pfld
│   └── pfld_mv2n_112.py
└── <Other Tasks...>
```

``` [按不同任务分类]
configs // [!code focus]
├── _base_ // [!code focus]
│   ├── datasets
│   │   └── coco_detection.py
│   ├── default_runtime_cls.py
│   ├── default_runtime_det.py
│   ├── default_runtime_pose.py
│   └── schedules
│       ├── schedule_1x.py
│       ├── schedule_20e.py
│       └── schedule_2x.py
├── fomo // [!code focus]
│   ├── fomo_efficientnet_b0_x8_abl_coco.py
│   ├── fomo_mobnetv2_0.35_x8_abl_coco.py
│   ├── fomo_mobnetv2_fpn_0.35_x8_abl.py
│   ├── fomo_mobnetv2_x8_coco.py
│   ├── fomo_mobnetv2_x8_voc.py
│   ├── fomo_mobnetv3_0.35_x8_abl_coco.py
│   ├── fomo_shufflenetv2_0.1_x8_abl_coco.py
│   └── fomo_squeezenet_0.1_x8_abl_coco.py
├── pfld // [!code focus]
│   └── pfld_mv2n_112.py
└── <Other Tasks...> // [!code focus]
```

``` [各任务中配置有不同的训练管线 (如 FOMO)]
configs // [!code focus]
├── _base_
│   ├── datasets
│   │   └── coco_detection.py
│   ├── default_runtime_cls.py
│   ├── default_runtime_det.py
│   ├── default_runtime_pose.py
│   └── schedules
│       ├── schedule_1x.py
│       ├── schedule_20e.py
│       └── schedule_2x.py
├── fomo // [!code focus]
│   ├── fomo_efficientnet_b0_x8_abl_coco.py // [!code focus]
│   ├── fomo_mobnetv2_0.35_x8_abl_coco.py // [!code focus]
│   ├── fomo_mobnetv2_fpn_0.35_x8_abl.py // [!code focus]
│   ├── fomo_mobnetv2_x8_coco.py // [!code focus]
│   ├── fomo_mobnetv2_x8_voc.py // [!code focus]
│   ├── fomo_mobnetv3_0.35_x8_abl_coco.py // [!code focus]
│   ├── fomo_shufflenetv2_0.1_x8_abl_coco.py // [!code focus]
│   └── fomo_squeezenet_0.1_x8_abl_coco.py // [!code focus]
├── pfld
│   └── pfld_mv2n_112.py
└── <Other Tasks...>
```

:::

::: tip

其中名为 `_base_` 的任务文件夹是我们其他任务的继承对象，关于配置文件继承的详细说明，请参考 [MMEngine - 配置文件的继承](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html#id3)。

:::


## 配置的内容结构

以 [FOMO 模型训练](./training/fomo.md)中的 `fomo_mobnetv2_0.35_x8_abl_coco.py` 配置文件为例，我们根据不同的功能模块介绍该配置文件中的各个字段。

### 重要参数

更改训练配置时，通常需要修改以下参数。例如，`height` 和 `width` 参数通常用于确定输入图像大小，其应与模型接受的输入尺寸保持一致，因此我们建议在配置文件中单独定义这些参数。

```python
height=96       # 输入图像高度
width=96        # 输入图像宽度
batch_size=16   # 验证期间单个 GPU 的批量大小
workers=4       # 验证期间单个 GPU 预读取数据的线程数
epoches=300     # 最大训练轮次: 300 轮
lr=0.001        # 学习率
```

### 网络模型

在 FOMO 模型的配置文件中，我们使用以下结构化的配置文件来设置检测算法组件，包括 Backbone、Neck 等重要的神经网络组件。部分模型配置如下:

```python
num_classes=2                         # 类别数
model=dict(
    type='Fomo',                      # 检测器名称
    backbone=dict(
        type='MobileNetV2',
        widen_factor=0.35,
        out_indices=(2, )),           # 主干网络配置
    head=dict(
        type='Fomo_Head',             # 检测头配置
        input_channels=16,            # 输入通道数，与 Neck 的输入通道一致
        num_classes=num_classes,      # 分类的类别数
        middle_channels=[96, 32],     # 检测头卷积的输出通道数
        act_cfg='ReLU6',              # 激活函数配置
        loss_cls=dict(
            type='BCEWithLogitsLoss', # Sigmoid 层和 BCELoss 结合的损失函数
            reduction='none',
            pos_weight=40),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
        cls_weight=40)                # pos_weight 的参数值
)
```

### 数据集和验证

在设置好网络模型后，我们还需要设置数据集和数据加载管道来构建数据加载器。由于这部分的复杂性，我们使用中间变量来简化数据加载器配置的编写。完整的数据增强方法可以在 `edgelab/datasets/pipelines` 文件夹中找到。

我们将在这里演示 FOMO 的训练和测试管线，该管线使用了[自定义的 COCO_MASK 数据集](./datasets/):

```python
dataset_type='FomoDatasets'   # 数据集类型，用于定义数据集
data_root=''                  # 数据根目录 (需要手动指定)
train_pipeline=[              # 训练数据加载管道
    dict(type='RandomResizedCrop', height=height, width=width, scale=(0.90, 1.1),
         p=1),                                 # RandomResizedCrop 数据增强
    dict(type='Rotate', limit=20),             # 最大为 20 度的随机旋转变换
    dict(type='RandomBrightnessContrast',      # RandomBrightnessContrast 数据增强
         brightness_limit=0.2,                 # 亮度改变的系数范围
         contrast_limit=0.2,                   # 对比度改变的系数范围
         p=0.5),                               # 使用 RandomBrightnessContrast 数据增强的概率
    dict(type='HorizontalFlip', p=0.5)         # 围绕 Y 轴水平翻转
]
test_pipeline=[dict(type='Resize', height=height, width=width,
                    p=1)]                      # 将输入调整为给定的高度和宽度

data=dict(samples_per_gpu=batch_size,          # 训练期间单个 GPU 的批量大小
          workers_per_gpu=workers,             # 验证期间单个 GPU 预读取数据的线程数
          train_dataloader=dict(collate=True), # 是否生成小批次样本
          val_dataloader=dict(collate=True),
          train=dict(type=dataset_type,
                     data_root=data_root,
                     ann_file='annotations/person_keypoints_train2017.json',
                     img_prefix='train2017',   # 图片路径与上方标注文件路径
                     pipeline=train_pipeline),
          val=dict(type=dataset_type,
                   data_root=data_root,
                   test_mode=True,             # 使能数据集的测试模式以避免过滤标注或图像
                   ann_file='annotations/person_keypoints_val2017.json',
                   img_prefix='val2017',
                   pipeline=test_pipeline),
          test=dict(type=dataset_type,
                    data_root=data_root,
                    test_mode=True,
                    ann_file='annotations/person_keypoints_val2017.json',
                    img_prefix='val2017',
                    pipeline=test_pipeline))
```

此外，我们还需要设置一个评估器。评估器用于计算训练模型在验证和测试数据集上的精度指标，其的配置由一个或一系列指标配置组成:

```python
evaluation=dict(interval=1, metric=['mAP'], fomo=True) # 验证评估指标 mAP
find_unused_parameters=True
```

### 优化器

```python
optimizer=dict(type='Adam', lr=lr, weight_decay=0.0005)         # 具有基础学习率和权重衰减的Adam梯度下降优化器
optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2)) # 用于构建优化器 Hook 的配置
```

:::tip

关于 Hook 的更多应用细节，请参考 [MMEngine - Hook](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/hook.html)。

:::

### 配置文件继承

在目录 `config/_base_` 下包含默认的配置文件，由 `_base_` 中的组件组成的配置文件称为原始配置。

为了便于测试，我们建议使用者继承现有的配置文件。例如，在 FOMO 模型的训练配置文件中设置有 `_base_='../_base_/default_runtime_det.py'`，然后在继承文件的基础上，我们修改配置文件中的必要字段。

```python
_base_='../_base_/default_runtime_det.py'
checkpoint_config=dict(interval=5) # 配置设置检查点 Hook，保存间隔为 5
log_config=dict(                   # 配置注册记录器 Hook
            interval=150,          # 打印日志间隔
            hooks=[
                dict(type='TextLoggerHook', ndigits=4),       # TXT 文本日志
                dict(type='TensorboardLoggerHook', ndigits=4) # Tensorboard 日志
            ])                                                # 记录训练过程的日志
epochs=300
runner=dict(type='EpochBasedRunner',  # 使用的 runner 类型 (例如 IterBasedRunner 或者 EpochBasedRunner)
            max_epochs=epochs)        # runner 运行 max_epochs 次工作流，对于 IterBasedRunner 使用 max_iters
dist_params=dict(backend='nccl')      # 设置分布式训练的参数，也可以设置端口
log_level='INFO'                      # 日志等级
load_from=None                        # 从给定路径加载模型作为预训练模型，不会恢复训练
resume_from=None                      # 从给定路径恢复检查点，训练将从保存检查点的轮次恢复
workflow=[('train', 1)]               # runner 的工作流程
opencv_num_threads=1                  # 关闭 OpenCV 多线程降低内存占用
work_dir='./work_dirs'                # 用于保存当前实验的模型检查点和日志的目录
```


## EdgeLab 参数化配置

使用 EdgeLab 的 `tools/train.py` 或 `tools/test.py` 提交作业时，可以指定 `--cfg-options` 临时覆写配置。

::: tip

可以按照原始配置中字典键的顺序指定配置选项并更新字典链的配置键。例如 `--cfg-options data_root='./dataset/coco'` 更改数据集的数据根目录。

:::


## FAQs

- 不同模型的配置文件会有一定的差异,我如何理解?

    更多细节请参考 [MMDet Config](https://mmdetection.readthedocs.io/zh_CN/latest/tutorials/config.html)，[MMPose Config](https://mmpose.readthedocs.io/zh_CN/latest/tutorials/0_config.html) 和 [MMCls Config](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html)。
