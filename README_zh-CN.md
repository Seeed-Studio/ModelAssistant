# 基于 OpenMMLab 框架的网络模型库

[English](./README.md) | 简体中文

## 简介

这是一个基于OpenMMLab 框架开发一个适用于多种视觉任务主干网络的模板库。

利用 OpenMMLab 框架，我们可以轻松地开发一个新的主干网络，并利用 MMClassification，MMDetection 和 MMSegmentation 来在分类、检测和分割等任务上进行基准测试。

## 配置环境

运行环境需要 [PyTorch](https://pytorch.org/get-started/locally/) 和以下 OpenMMLab 仓库:

- [MIM](https://github.com/open-mmlab/mim): 一个用于管理 OpenMMLab 包和实验的命令行工具
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱和基准测试。除了分类任务，它同时用于提供多样的主干网络
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱和基准测试
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱和基准测试

假设你已经准备好了 Python 和 PyTorch 环境，那么只需要下面两行命令，就可以配置好软件环境。

```bash
pip install openmim mmcls mmdet mmsegmentation
mim install mmcv-full
```

## 数据准备

本仓库示例所使用的数据需要按照如下结构组织：

```text
data/
├── imagenet
│   ├── train
│   ├── val
│   └── meta
│       ├── train.txt
│       └── val.txt
├── ade
│   └── ADEChallengeData2016
│       ├── annotations
│       └── images
└── coco
    ├── annotations
    │   ├── instance_train2017.json
    │   └── instance_val2017.json
    ├── train2017
    └── val2017
```

这里，我们只列举了用于训练和验证 ImageNet（分类任务）、ADE20K（分割任务）和 COCO（检测任务）的必要文件。

如果你希望在更多数据集或任务上进行基准测试，比如使用 MMDetection 进行全景分割，只需要按照 MMDetection 
的需要组织对应的数据集即可。对于语义分割任务，也可以参照这篇[教程](https://mmsegmentation.readthedocs.io/zh_CN/latest/dataset_prepare.html)组织数据集。

## 用法

### 实现你的主干网络

在这个示例库中，我们以 ConvNeXt 为例展示如何快速实现一个主干网络。

1. 在 `models` 文件夹下创建你的主干网络文件。例如本例中的 [`models/convnext.py`](models/convnext.py)。

   在这个文件中，只需要使用 PyTorch 实现你的主干网络即可，额外需要注意的只有以下两点：

   1. 主干网络和主要模块需要继承 `mmcv.runner.BaseModule`。这个 `BaseModule` 是 `torch.nn.Module` 的子类，
      其行为几乎完全一致，除了它额外支持使用 `init_cfg` 参数指定包括预训练模型在内的初始化方法。

   2. 使用如下一行装饰器将你的主干网络类注册至 `mmcls.models.BACKBONES` 注册器。
      ```python
      @BACKBONES.register_module(force=True)
      ```
      > :question: 注册器是什么？看看[这个](https://mmcv.readthedocs.io/zh_CN/latest/understand_mmcv/registry.html)！

2. **[可选]** 如果你希望为某些特定的任务增加一些额外的模块、组件，可以参照 [`models/det/layer_decay_optimizer_constructor.py`](models/det/layer_decay_optimizer_constructor.py) 添加至对应的文件夹中。

3. 将你添加的主干网络、自定义组件添加至 [`models/__init__.py`](models/__init__.py) 文件中。

### 添加配置文件

将每个任务对应的配置文件放在 [`configs/`](./configs)。如果你不太清楚配置文件是怎么回事，这篇[教程](https://mmclassification.readthedocs.io/zh_CN/latest/tutorials/config.html#config-file-structure)应该可以帮到你。

简而言之，使用若干子配置文件来组织你的配置文件，这些子配置文件一般包括模型、数据集、优化策略和运行配置。你也可以在配置文件中对子配置文件进行覆盖，或者不使用子配置文件，全部写到一个文件里。

在本示例库中，我们提供了一套比较常用的子配置文件，你可以在从以下位置找到更多有用的子配置文件：
[mmcls](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_)、
[mmdet](https://github.com/open-mmlab/mmdetection/tree/master/configs/_base_)、
[mmseg](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/_base_)。

### 训练和测试

要进行训练和测试，可以直接使用 mim 工具

首先，我们需要将当前仓库的目录添加到 `PYTHONPATH` 环境变量中，这样 Python 才可以找到我们的模型文件。

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH 
```

#### 单机单 GPU：

```bash
# 训练分类模型
mim train mmcls $CONFIG --work-dir $WORK_DIR

# 测试分类模型
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)"

# 训练目标检测/实例分割模型
mim train mmdet $CONFIG --work-dir $WORK_DIR

# 测试目标检测/实例分割模型
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm

# 训练语义分割模型
mim train mmseg $CONFIG --work-dir $WORK_DIR

# 测试语义分割模型
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU
```

- CONFIG: `configs/` 文件夹中的配置文件。
- WORK_DIR: 用于保存日志和权重文件的文件夹
- CHECKPOINT: 权重文件路径

#### 单机多 GPU （以 4 GPU 为例）：

```bash
# 训练分类模型
mim train mmcls $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4

# 测试分类模型
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)" --launcher pytorch --gpus 4

# 训练目标检测/实例分割模型
mim train mmdet $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4

# 测试目标检测/实例分割模型
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm --launcher pytorch --gpus 4

# 训练语义分割模型
mim train mmseg $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4 

# 测试语义分割模型
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU --launcher pytorch --gpus 4
```

- CONFIG: `configs/` 文件夹中的配置文件。
- WORK_DIR: 用于保存日志和权重文件的文件夹
- CHECKPOINT: 权重文件路径

#### 多机多 GPU （以 2 节点共计 16 GPU 为例）

```bash
# 训练分类模型
mim train mmcls $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 测试分类模型
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)" --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 训练目标检测/实例分割模型
mim train mmdet $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 测试目标检测/实例分割模型
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 训练语义分割模型
mim train mmseg $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 测试语义分割模型
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

- CONFIG: `configs/` 文件夹中的配置文件。
- WORK_DIR: 用于保存日志和权重文件的文件夹
- CHECKPOINT: 权重文件路径
- PARTITION: 使用的 Slurm 分区
