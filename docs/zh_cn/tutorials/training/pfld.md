# PFLD 模型训练

本节将介绍如何在 PFLD 表计数据集上训练 PFLD 表计模型。PFLD 模型是在论文 [PFLD: A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf) 中提出的。

## 准备数据集

在开始训练前，首先需要准备标注好的数据集。EdgeLab 默认使用[自定义 Meter 数据集](../datasets.md#EdgeLab)训练 PFLD 模型，请参照以下步骤完成数据集的准备。

1. 参考[互联网数据集 - EdgeLab - 自定义 Meter 数据集](../datasets.md#EdgeLab)下载数据集并完成数据集的解压。

2. 记住数据集解压后的**文件夹路径** (如: `datasets\meter`)，在之后修改配置文件时需要使用该文件夹路径。

## 选择配置文件

我们将根据需要执行的训练任务类型来选择合适的配置文件，我们已经在[模型配置](../config.md)中对配置文件的功能、结构、原理进行了简单介绍。

对于表计 PFLD 模型示例，我们使用 `pfld_mbv2n_112.py` 作为配置文件，它位于 EdgeLab 主目录路径 `configs/pfld` 下的文件夹中，并额外继承了 `default_runtime_pose.py` 配置文件。

配置文件内容如下，对于初学者，我们建议首先注意该配置文件中 `data_root` 和 `epochs` 这两个参数。

::: details `pfld_mbv2n_112.py`

```python
_base_='../_base_/default_runtime_pose.py'

num_classes=1
model=dict(type='PFLD',
             backbone=dict(type='PfldMobileNetV2',
                           inchannel=3,
                           layer1=[16, 16, 16, 16, 16],
                           layer2=[32, 32, 32, 32, 32, 32],
                           out_channel=16),
             head=dict(type='PFLDhead',
                       num_point=num_classes,
                       input_channel=16,
                       loss_cfg=dict(type='L1Loss')))

# dataset settings
dataset_type='MeterData'

data_root=''
height=112
width=112
batch_size=32
workers=4

train_pipeline=[
    dict(type="Resize", height=height, width=width, interpolation=0),
    dict(type='ColorJitter', brightness=0.3, p=0.5),
    dict(type='GaussNoise'),
    dict(type='MedianBlur', blur_limit=3, p=0.3),
    dict(type='HorizontalFlip'),
    dict(type='VerticalFlip'),
    dict(type='Rotate'),
    dict(type='Affine', translate_percent=[0.05, 0.1], p=0.6)
]

val_pipeline=[dict(type="Resize", height=height, width=width)]

train_dataloader=dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 index_file=r'train/annotations.txt',
                 pipeline=train_pipeline,
                 test_mode=False),
)

val_dataloader=dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 index_file=r'val/annotations.txt',
                 pipeline=val_pipeline,
                 test_mode=True),
)
test_dataloader=val_dataloader

lr=0.0001
epochs=300
evaluation=dict(save_best='loss')
optim_wrapper=dict(
    optimizer=dict(type='Adam', lr=lr, betas=(0.9, 0.99), weight_decay=1e-6))
optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))
val_evaluator=dict(type='PointMetric')
test_evaluator=val_evaluator
find_unused_parameters=True
train_cfg=dict(by_epoch=True, max_epochs=500)

# learning policy
param_scheduler=[
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(type='MultiStepLR',
         begin=1,
         end=500,
         milestones=[350, 400, 450, 490],
         gamma=0.1,
         by_epoch=True)
]
```

:::

## 训练模型

训练模型需要使用我们之前配置好的 EdgeLab 工作环境，如果您按照我们的[安装指南](../../introduction/installation.md)使用 Conda 将 EdgeLab 安装在了名为 `edgelab` 的虚拟环境中，请首先确保您目前正处在虚拟环境中。

然后，在 EdgeLab 项目根目录，我们执行如下的指令，训练一个端到端的表计 PFLD 模型。

```sh
python3 tools/train.py \
    configs/pfld/pfld_mbv2n_112.py \
    --cfg-options \
        data_root='datasets/meter' \
        epochs=50
```

在训练期间，训练得到的模型权重和相关的日志信息会默认保存至路径 `work_dirs/pfld_mbv2n_112` 下，您可以使用 [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) 等工具事实监测训练情况。

```sh
tensorboard --logdir work_dirs/pfld_mbv2n_112
```

在训练完成后，最新的 PFLD 模型权重文件的路径会保存在 `work_dirs/pfld_mbv2n_112/last_checkpoint` 文件中。请留意权重文件路径，在将模型转换为其他格式时需要用到。

::: tip

如果您配置了虚拟环境但并未激活，您可以使用以下命令激活虚拟环境:

```sh
conda activate edgelab
```

:::

## 测试和评估

### 测试

在完成了 PFLD 模型的训练后，您可以使用以下命令，指定特定权重并对模型进行简单测试:

```sh
python3 tools/inference.py \
    configs/pfld/pfld_mbv2n_112.py \
    "$(cat work_dirs/pfld_mbv2n_112/last_checkpoint)" \
    --cfg-options \
        data_root='datasets/meter'
```

::: tip

如果您在测试时希望实时的可视化预览，您可以在测试命令后追加一个参数 `--show` 来显示预测结果。对于更多可选的参数，请参考源代码 `tools/test.py`。

:::

### 评估

为了进一步在现实中的边缘计算设备上测试和评估模型，您需要导出模型。在导出模型的过程中，EdgeLab 会对模型进行一些优化，如模型的剪枝、蒸馏等。您可以参考[模型导出](../export/overview)章节进一步学习如何导出模型。
