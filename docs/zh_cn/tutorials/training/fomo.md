# FOMO 模型训练

本节将介绍如何在 COCO MASK 数据集上训练 FOMO 口罩检测模型。FOMO 口罩检测模型基于 MobileNet V2 和 MobileNet V3 实现 (实际选用的神经网络取决于您选择的模型配置文件)。

关于 MobileNet 的更多信息，请参考论文 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)。

## 准备数据集

[SSCMA](https://github.com/Seeed-Studio/SSCMA) 默认使用 [COCO_MASK 数据集](../datasets.md#SSCMA)训练 FOMO 模型，请参照以下步骤完成数据集的准备。

1. 参考[互联网数据集 - SSCMA - COCO_MASK 数据集](../datasets.md#SSCMA)下载数据集并完成数据集的解压。

2. 记住数据集解压后的**文件夹路径** (如: `datasets\mask`)，在之后修改配置文件时需要使用该文件夹路径。

## 选择配置文件

我们将根据需要执行的训练任务类型来选择合适的配置文件，我们已经在[模型配置](../config.md)中对配置文件的功能、结构、原理进行了简单介绍。

对于 FOMO 模型示例，我们使用 `fomo_mobnetv2_0.35_x8_abl_coco.py` 作为配置文件，它位于 [SSCMA](https://github.com/Seeed-Studio/SSCMA) 主目录路径 `configs/fomo` 下的文件夹中，并额外继承了 `default_runtime_det.py` 配置文件。

配置文件内容如下，对于初学者，我们建议首先注意该配置文件中 `data_root` 和 `epochs` 这两个参数。

::: details `fomo_mobnetv2_0.35_x8_abl_coco.py`

```python
_base_='../_base_/default_runtime_det.py'
default_scope='sscma'
custom_imports=dict(imports=['sscma'], allow_failed_imports=False)

num_classes=2
model=dict(type='Fomo',
           backbone=dict(type='mmdet.MobileNetV2', widen_factor=0.35, out_indices=(2,)),
           head=dict(type='FomoHead',
                     input_channels=[16],
                     num_classes=num_classes,
                     middle_channel=48,
                     act_cfg='ReLU6',
                     loss_cls=dict(type='BCEWithLogitsLoss',
                                   reduction='none',
                                   pos_weight=40),
                     loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
           ),
)

# dataset settings
dataset_type='FomoDatasets'
data_root=''
height=96
width=96
batch_size=16
workers=1

train_pipeline=[
    dict(type='RandomResizedCrop',
         height=height,
         width=width,
         scale=(0.80, 1.2),
         p=1),
    dict(type='Rotate', limit=30),
    dict(type='RandomBrightnessContrast',
         brightness_limit=0.3,
         contrast_limit=0.3,
         p=0.5),
    dict(type='HorizontalFlip', p=0.5),
]
test_pipeline=[dict(type='Resize', height=height, width=width, p=1)]

train_dataloader=dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='fomo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 ann_file='train/_annotations.coco.json',
                 img_prefix='train',
                 pipeline=train_pipeline),
)
val_dataloader=dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='fomo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 ann_file='valid/_annotations.coco.json',
                 img_prefix='valid',
                 pipeline=test_pipeline))
test_dataloader=val_dataloader

# optimizer
lr=0.001
epochs=300
find_unused_parameters=True
optim_wrapper=dict(optimizer=dict(type='Adam', lr=lr, weight_decay=5e-4,eps=1e-7))

#evaluator
val_evaluator=dict(type='FomoMetric')
test_evaluator=val_evaluator
train_cfg=dict(by_epoch=True,max_epochs=70)

# learning policy
param_scheduler=[
    dict(type='LinearLR', begin=0, end=30, start_factor=0.001, by_epoch=False),  # warm-up
    dict(type='MultiStepLR',
         begin=1,
         end=500,
         milestones=[100, 200, 250],
         gamma=0.1,
         by_epoch=True)
]
```

:::

## 训练模型

训练模型需要使用我们之前配置好的 [SSCMA](https://github.com/Seeed-Studio/SSCMA)  工作环境，如果您按照我们的[安装指南](../../introduction/installation.md)使用 Conda 将 [SSCMA](https://github.com/Seeed-Studio/SSCMA)  安装在了名为 `sscma` 的虚拟环境中，请首先确保您目前正处在虚拟环境中。

然后，在 [SSCMA](https://github.com/Seeed-Studio/SSCMA)  项目根目录，我们执行如下的指令，训练一个 FOMO 口罩识别模型。

```sh
python3 tools/train.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    --cfg-options \
        data_root='datasets/mask' \
        epochs=50
```

在训练期间，训练得到的模型权重和相关的日志信息会默认保存至路径 `work_dirs/fomo_mobnetv2_0.35_x8_abl_coco` 下，您可以使用 [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) 等工具事实监测训练情况。

```sh
tensorboard --logdir work_dirs/fomo_mobnetv2_0.35_x8_abl_coco
```

在训练完成后，最新的 FOMO 模型权重文件的路径会保存在 `work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint` 文件中。请留意权重文件路径，在将模型转换为其他格式时需要用到。

::: tip

如果您配置了虚拟环境但并未激活，您可以使用以下命令激活虚拟环境:

```sh
conda activate sscma
```

:::

## 测试和评估

### 测试

在完成了 FOMO 模型的训练后，您可以使用以下命令，指定特定权重并测试模型:

```sh
python3 tools/inference.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint)" \
    --cfg-options \
        data_root='datasets/mask'
```

::: tip

如果您在测试时希望实时的可视化预览，您可以在测试命令后追加一个参数 `--show` 来显示预测结果。对于更多可选的参数，请参考源代码 `tools/test.py`。

:::

### 评估

为了进一步在现实中的边缘计算设备上测试和评估模型，您需要导出模型。在导出模型的过程中，[SSCMA](https://github.com/Seeed-Studio/SSCMA) 会对模型进行一些优化，如模型的剪枝、蒸馏等。您可以参考[模型导出](../export/overview)章节进一步学习如何导出模型。

### 部署

在导出模型后，你可以将模型部署到边缘计算设备上进行测试和评估。你可以参考 [Deploy](./../../deploy/overview.md) 部分来了解更多关于如何部署模型的信息。
