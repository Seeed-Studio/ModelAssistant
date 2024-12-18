# FOMO 模型训练

本节将介绍如何在 COCO MASK 数据集上训练 FOMO 口罩检测模型。FOMO 口罩检测模型基于 MobileNet V2 和 MobileNet V3 实现 (实际选用的神经网络取决于您选择的模型配置文件)。关于 MobileNet 的更多信息，请参考论文 [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)。


## 数据集准备

在进行 FOMO 模型训练之前，我们需要准备好数据集。这里我们以已经标注好的口罩 COCO 数据集为例，您可以在 [SSCMA - 公共数据集](../../datasets/public#获取公共数据集) 中下载该数据集。


## 模型选择与训练

SSCMA 提供了多种不同的 FOMO 模型配置，您可以根据自己的需求选择合适的模型进行训练。

```sh
fomo_mobnetv2_0.1_x8_coco.py
fomo_mobnetv2_0.35_x8_coco.py
fomo_mobnetv2_1_x16_coco.py
```

在此我们以 `fomo_mobnetv2_0.35_x8_coco.py` 为例，展示如何使用 SSCMA 进行 FOMO 模型训练。

```sh
python3 tools/train.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_coco.py \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/ \
    epochs=50 \
    height=192 \
    width=192
```

- `configs/fomo/fomo_mobnetv2_0.35_x8_coco.py`: 指定配置文件，定义模型和训练设置。
- `--cfg-options`: 用于指定额外的配置选项。
    - `data_root`: 设定数据集的根目录。
    - `num_classes`: 指定模型需要识别的类别数量。
    - `train_ann`: 指定训练数据的注释文件路径。
    - `val_ann`: 指定验证数据的注释文件路径。
    - `train_data`: 指定训练图像的前缀路径。
    - `val_data`: 指定验证图像的前缀路径。
    - `epochs`: 设置训练的最大周期数。

等待训练结束后，您可以在 `work_dirs/fomo_mobnetv2_0.35_x8_coco` 目录下找到训练好的模型，在查找模型前，我们建议先关注训练结果。以下是对结果的分析以及一些改进方向。


:::details

```sh
12/18 01:47:05 - mmengine - INFO - Epoch(val) [50][6/6]    P: 0.2545  R: 0.4610  F1: 0.3279  data_time: 0.0644  time: 0.0798
```

F1 综合了精确率（Precision）和召回率（Recall）两个指标，旨在提供一个单一的数字来衡量模型的整体性能,F1 分数的值范围在 0 到 1 之间，值越高表示模型的精确率和召回率都越高，性能越好。当模型的精确率和召回率相等时，F1 分数达到最大值。

:::

## 模型导出及验证

在训练过程中，您可以随时查看训练日志、导出模型并验证模型的性能，部分模型验证中输出的指标在训练过程中也会显示，因此在这一部分我们会先介绍如何导出模型，然后阐述如何验证导出后模型的精度。

### 导出模型

这里我们以导出 TFLite 模型为例，您可以使用以下命令导出不同精度的 TFLite 模型：

```sh
python3 tools/export.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_coco.py \
    work_dirs/fomo_mobnetv2_0.35_x8_coco/epoch_50.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/ \
    epochs=50 \
    --imgsz 192 192 \
    --format tflite \
    --image_path $(pwd)/datasets/coco_mask/mask/valid
```

:::warning

我们建议在训练和导出时使用相同的分辨率，在当前情况下，使用不同的分辨率训练和导出时，可能导致模型精度降低或完全丢失。

:::

:::tip

在导出过程中，可能需要网络环境以安装某些依赖，如果无法访问到互联网，请确保当前的 Python 环境中已经安装了以下依赖：

```
tensorflow
hailo_sdk_client
onnx
onnx2tf
tf-keras
onnx-graphsurgeon
sng4onnx
onnxsim
```

此外，`onnx2tf` 在运行时可能还需要下载 calibration 相关的数据，您可以参考以下链接将其提前下载到 SSCMA 的根目录。

```sh
wget https://github.com/PINTO0309/onnx2tf/releases/download/1.20.4/calibration_image_sample_data_20x128x128x3_float32.npy \
    -O calibration_image_sample_data_20x128x128x3_float32.npy
```

:::


### 验证模型

在导出模型后，您可以使用以下命令验证模型的性能：

```sh
python3 tools/test.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_coco.py \
    work_dirs/fomo_mobnetv2_0.35_x8_coco/epoch_50_int8.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/ \
    height=192 \
    width=192
```

### QAT

QAT（量化感知训练）是一种在模型训练过程中模拟量化操作，让模型逐步适应量化误差，从而在量化后仍能保持较高精度的方法。SSCMA 支持 QAT，您可以参考以下方法得到 QAT 的模型，并再次验证。

```sh
python3 tools/quantization.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_coco.py \
    work_dirs/fomo_mobnetv2_0.35_x8_coco/epoch_50.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/ \
    epochs=5 \
    height=192 \
    width=192
```

QAT 训练完毕后，会自动导出量化后的模型，其存放路径为 `out/qat_model_test.tflite`，您可以使用以下命令对其进行验证：

```sh
python3 tools/test.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_coco.py \
    work_dirs/fomo_mobnetv2_0.35_x8_coco/qat/qat_model_int8.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/ \
    height=192 \
    width=192
```

