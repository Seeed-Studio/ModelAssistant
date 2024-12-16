# 基础配置结构

SSCMA 使用的配置文件位于 `configs` 目录下，用于不同任务下不同模型的训练。我们在其根据不同的任务分类划分了子文件夹，在各个子文件夹中，保存有多个模型的不同训练管线参数。

:::tip

其中名为 `_base_` 的任务文件夹是我们其他任务的继承对象，关于配置文件继承的详细说明，请参考 [MMEngine - 配置文件的继承](https://mmengine.readthedocs.io/zh_CN/latest/advanced_tutorials/config.html#id3)。

:::

我们使用 Python 的字典和列表来定义模型、数据加载、训练和评估等各个部分的参数。以下是该配置文件的结构和各个部分的作用，以及一般需要调整的参数。

## 导入模块

在开始构建你的配置文件之前，首先需要导入必要的模块，如下所示：

```python
import torch.nn as nn
from mmengine.registry import OPTIMIZERS
```

## 模型配置

- `num_classes`：类别数量，对于口罩检测，通常是 2（戴口罩和不戴口罩）。
- `widen_factor`：模型宽度因子，用于调整模型的宽度。

## 数据配置

- `dataset_type`：指定数据集类型。
- `data_root`：数据集的根目录。
- `train_ann`、`train_data`、`val_ann`、`val_data`：训练和验证数据的注释文件和数据目录。
- `height`、`width`、`imgsz`：输入图像的尺寸。

## 训练配置

- `batch`、`workers`、`persistent_workers`：训练时的批处理大小、工作线程数和持久工作线程。
- `val_batch`、`val_workers`：验证时的批处理大小和工作线程数。
- `lr`、`epochs`、`weight_decay`、`momentum`：学习率、训练周期、权重衰减和动量。

## 钩子

- `default_hooks`：定义了训练过程中的钩子，如可视化钩子。
- `visualizer`：定义了可视化器。

## 数据预处理

- `data_preprocessor`：定义了数据预处理的参数，如均值、标准差、颜色转换等。

## 模型结构

定义了模型的类型、数据预处理器、骨干网络和头部网络的配置。

## 部署配置

- `deploy`：定义了模型部署时的数据预处理器配置。

## 图像解码后端

- `imdecode_backend`：指定图像解码的后端。

## 预处理流水线

- `pre_transform`、`train_pipeline`、`test_pipeline`：定义了训练和测试数据的预处理流水线。

## 数据加载器

- `train_dataloader`、`val_dataloader`、`test_dataloader`：定义了训练、验证和测试数据加载器的配置。

## 优化器配置

- `optim_wrapper`：定义了优化器的类型和参数。

## 评估器

- `val_evaluator`、`test_evaluator`：定义了验证和测试的评估器。

## 训练配置

- `train_cfg`：定义了训练的配置，如是否按周期训练和最大周期数。

## 学习策略

- `param_scheduler`：定义了学习率调度器的策略。

配置文件涵盖了从数据预处理到模型训练和评估的各个方面。根据具体的训练需求，可能需要调整的参数包括学习率、批次大小、训练周期、优化器参数、数据增强策略等。这些参数的调整将直接影响模型的性能和训练效果。