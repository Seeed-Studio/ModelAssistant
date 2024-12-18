# PFLD 模型训练

本节将介绍如何在 PFLD 表计数据集上训练 PFLD 表计模型。PFLD 模型是在论文 [PFLD: A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf) 中提出的。


## 数据集准备

在进行 PFLD 模型训练之前，我们需要准备好数据集。这里我们以已经标注好的 Meter 数据集为例，您可以在 [SSCMA - 公共数据集](../../datasets/public#获取公共数据集) 中下载该数据集。


## 模型选择与训练

SSCMA 提供了多种不同的 FOMO 模型配置，您可以根据自己的需求选择合适的模型进行训练。

```sh
pfld_mbv2_1000e.py
pfld_mbv3l_192_1000e.py
```

在此我们以 `pfld_mbv2_1000e.py` 为例，展示如何使用 SSCMA 进行 FOMO 模型训练。

```sh
python3 tools/train.py \
    configs/pfld/pfld_mbv2_1000e.py \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    epochs=100 \
    val_workers=2
```

- `configs/pfld/pfld_mbv2_1000e.py`: 指定配置文件，定义模型和训练设置。
- `--cfg-options`: 用于指定额外的配置选项。
    - `data_root`: 设定数据集的根目录。
    - `epochs`: 设置训练的最大周期数。


等待训练结束后，您可以在 `work_dirs/pfld_mbv2_1000e` 目录下找到训练好的模型。

:::details

```sh
12/16 06:40:25 - mmengine - INFO - Exp name: pfld_mbv2_1000e_20241216_062913
12/16 06:40:25 - mmengine - INFO - Saving checkpoint at 100 epochs
12/16 06:40:25 - mmengine - INFO - Epoch(val) [100][30/30]    keypoint/Acc: 0.8538  data_time: 0.0116  time: 0.0227
```

在训练过程中，您可以查看训练日志，以及训练过程中的指标，如关键点准确率等。

:::

:::tip

如果您配置了虚拟环境但并未激活，您可以使用以下命令激活虚拟环境:

```sh
conda activate sscma
```

:::

## 模型导出及验证

在训练过程中，您可以随时查看训练日志、导出模型并验证模型的性能，部分模型验证中输出的指标在训练过程中也会显示，因此在这一部分我们会先介绍如何导出模型，然后阐述如何验证导出后模型的精度。

### 导出模型

这里我们以导出 TFLite 模型为例，您可以使用以下命令导出不同精度的 TFLite 模型：

```sh
python3 tools/export.py \
    configs/pfld/pfld_mbv2_1000e.py \
    work_dirs/pfld_mbv2_1000e/epoch_100.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    val_workers=2 \
    --imgsz 112 112 \
    --format tflite \
    --image_path $(pwd)/datasets/meter/val/images
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
    configs/pfld/pfld_mbv2_1000e.py \
    work_dirs/pfld_mbv2_1000e/epoch_100_int8.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    val_workers=2 
```

### QAT

QAT（量化感知训练）是一种在模型训练过程中模拟量化操作，让模型逐步适应量化误差，从而在量化后仍能保持较高精度的方法。SSCMA 支持 QAT，您可以参考以下方法得到 QAT 的模型，并再次验证。

```sh
python3 tools/quantization.py \
    configs/pfld/pfld_mbv2_1000e.py \
    work_dirs/pfld_mbv2_1000e/epoch_100.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    epochs=5 \
    val_workers=2
```

QAT 训练完毕后，会自动导出量化后的模型，您可以使用以下命令对其进行验证：

```sh
python3 tools/test.py \
    configs/pfld/pfld_mbv2_1000e.py \
    work_dirs/rtmdet_nano_8xb32_300e_coco/qat/qat_model_int8.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/meter/ \
    val_workers=2  
```
