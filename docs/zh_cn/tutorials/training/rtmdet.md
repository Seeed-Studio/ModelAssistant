# RTMDet 模型训练

RTMDet (Real-time Models for Object Detection) 是一个高精度、低延时的单阶段目标检测器算法，RTMDet 模型整体结构和 YOLOX 几乎一致，由 CSPNeXt + CSPNeXtPAFPN + 共享卷积权重但分别计算 BN 的 SepBNHead 构成。内部核心模块也是 CSPLayer，但对其中的 Basic Block 改进为了 CSPNeXt Block。

## 数据集准备

在进行 RTMDet 模型训练之前，我们需要准备好数据集。这里我们以已经标注好的口罩 COCO 数据集为例，您可以在 [SSCMA - 公共数据集](../../datasets/public#获取公共数据集) 中下载该数据集。

## 模型选择与训练

SSCMA 提供了多种不同的 RTMDet 模型配置，您可以根据自己的需求选择合适的模型进行训练。

```sh
rtmdet_l_8xb32_300e_coco.py
rtmdet_m_8xb32_300e_coco.py
rtmdet_mnv4_8xb32_300e_coco.py
rtmdet_nano_8xb32_300e_coco.py
rtmdet_nano_8xb32_300e_coco_relu.py
rtmdet_nano_8xb32_300e_coco_relu_q.py
rtmdet_s_8xb32_300e_coco.py
```

在此我们以 `rtmdet_nano_8xb32_300e_coco.py` 为例，展示如何使用 SSCMA 进行 RTMDet 模型训练。

```sh
python3 tools/train.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    max_epochs=150 \
    imgsz='(192,192)'
```

- `configs/rtmdet_nano_8xb32_300e_coco.py`: 指定配置文件，定义模型和训练设置。
- `--cfg-options`: 用于指定额外的配置选项。
    - `data_root`: 设定数据集的根目录。
    - `num_classes`: 指定模型需要识别的类别数量。
    - `train_ann_file`: 指定训练数据的注释文件路径。
    - `val_ann_file`: 指定验证数据的注释文件路径。
    - `train_img_prefix`: 指定训练图像的前缀路径。
    - `val_img_prefix`: 指定验证图像的前缀路径。
    - `max_epochs`: 设置训练的最大周期数。
    - `imgsz`：指定模型训练使用的图像尺寸。

等待训练结束后，您可以在 `work_dirs/rtmdet_nano_8xb32_300e_coco` 目录下找到训练好的模型，在查找模型前，我们建议先关注训练结果。以下是对结果的分析以及一些改进方向。

:::details

```sh
12/17 03:55:23 - mmengine - INFO - Saving checkpoint at 150 epochs
12/17 03:55:24 - mmengine - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.31s).
Accumulating evaluation results...
DONE (t=0.06s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.506
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.946
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.456
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.517
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.591
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.100
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
12/17 03:55:24 - mmengine - INFO - bbox_mAP_copypaste: 0.506 0.946 0.456 -1.000 0.000 0.517
12/17 03:55:24 - mmengine - INFO - Epoch(val) [150][6/6]    coco/bbox_mAP: 0.5060  coco/bbox_mAP_50: 0.9460  coco/bbox_mAP_75: 0.4560  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: 0.0000  coco/bbox_mAP_l: 0.5170  data_time: 0.0205  time: 0.0563
```

通过分析 COCO Eval 的结果可以发现问题所在，并采取相应的措施进行优化，优化方向建议优先从数据集入手，再是训练参数、以及模型结构。

平均精度（AP）方面：
- 在 IoU=0.50:0.95 且 area=all 时，AP 为 0.506，整体处于中等偏低水平，模型在不同交并比综合情况下的检测精度有一定提升空间。
- 当 IoU=0.50 时 AP 达到 0.946，表明在较宽松交并比要求下模型能有较好表现，但 IoU=0.75 时 AP 仅 0.456，意味着模型在高交并比情况下精度很差，对预测框和真实框重合度要求高时表现不佳。
- 按检测目标面积分类来看，area=small 时 AP 为 -1.000，数据异常，小目标检测存在严重问题，数据集的验证集缺乏小目标。而 area=medium 时 AR 和 AP 为 0，表明验证集中包含中等目标的目标，但存在一些其他问题，如训练集中缺乏中等目标、数据增强参数异常等。

平均召回率（AR）方面：
- 在 IoU=0.50:0.95 且 area=all 的不同 maxDets 下，随着 maxDets 从 1 增加到 100，AR 从 0.547 提升到 0.608，增加可检测目标数量上限能在一定程度上提高召回率，但整体数值不算高，模型在实际情况下可能遗漏较多目标。
- 按面积分类中，area=small 的 AR 为 -1.000，再次体现数据集的验证集缺乏小目标的问题。

根据以上数据，我们首先检查数据集中是否由足够的小目标物体、小目标的数据标注是否准确、完整，必要时重新进行标注，确保标注框贴合小目标实际边界，再检查数据集通过训练管线后，经过数据增强后的图像色彩以及标注是否正确、合理。

此外，我们还需要检查训练过程，模型是否收敛等，您可以使用 Tensorboard 来进行查看。

安装并运行时 Tensorboard：

```sh
python3 -m pip install tensorboard && \
    tensorboard --logdir workdir
```
在 Scalars 选项卡下，可以查看记录的标量指标（如损失、准确率）随时间（通常是训练轮数）的变化情况。通过观察损失函数的下降趋势和准确率的上升趋势，判断模型是否在正常收敛。如果损失函数不再下降或者准确率不再上升，可能表示模型已经收敛或者出现了问题，这里仅简单介绍一下调整策略。

- **学习率**：如果损失函数下降过于缓慢，可以尝试增大学习率；如果损失函数出现剧烈震荡或者不收敛，可能是学习率过大，需要减小学习率。对于学习率的调整策略等请参考 [SSCMA - 自定义 - 基础配置结构](../../custom/basics.md)
- **迭代次数**：如果模型在训练过程中还没有完全收敛（例如损失函数仍在下降，准确率仍在上升），可以适当增加迭代次数。如果模型已经收敛，继续增加迭代次数可能会导致过拟合，此时可以减少迭代次数。

:::

在 `work_dirs/rtmdet_nano_8xb32_300e_coco` 目录下找到训练好的模型。此外，当模型训练结果精度不佳时，通过分析 COCO Eval 的结果可以发现问题所在，并采取相应的措施进行优化。

:::tip

当模型训练结果精度不佳时，通过分析 COCO Eval 的结果可以发现问题所在，并采取相应的措施进行优化。

:::


## 模型导出及验证

在训练过程中，您可以随时查看训练日志、导出模型并验证模型的性能，部分模型验证中输出的指标在训练过程中也会显示，因此在这一部分我们会先介绍如何导出模型，然后阐述如何验证导出后模型的精度。

### 导出模型

这里我们以导出 TFLite 模型为例，您可以使用以下命令导出不同精度的 TFLite 模型：

```sh
python3 tools/export.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    work_dirs/rtmdet_nano_8xb32_300e_coco/epoch_150.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
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

导出完成后，您可以使用以下命令对 TFLite Int8 模型进行验证：

```sh
python3 tools/test.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    work_dirs/rtmdet_nano_8xb32_300e_coco/epoch_150_int8.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    imgsz='(192,192)'
```

得到以下输出:

```sh
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.046
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.112
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.019
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.050
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.106
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.165
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.200
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.353
```

从验证结果可以看出，导出的模型在验证集上的表现与训练时的表现有所差异，AP@50:95 相比训练时下降了 46.0%，AP@50 下降了 83.4%，您可以尝试使用 QAT 来减少量化精度的损失。


:::tip

关于以上输出的详细解释，请参考 [COCO 数据集评估指标](https://cocodataset.org/#detection-eval)，在这里我们主要关注 50-95 IoU 和 50 IoU 的 mAP。

:::


### QAT

QAT（量化感知训练）是一种在模型训练过程中模拟量化操作，让模型逐步适应量化误差，从而在量化后仍能保持较高精度的方法。SSCMA 支持 QAT，您可以参考以下方法得到 QAT 的模型，并再次验证。

```sh
python3 tools/quantization.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    work_dirs/rtmdet_nano_8xb32_300e_coco/epoch_150.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    imgsz='(192,192)' \
    max_epochs=5
```

:::details

QAT 训练结果：

```sh
12/17 09:43:41 - mmengine - INFO - Saving checkpoint at 5 epochs
12/17 09:43:43 - mmengine - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.31s).
Accumulating evaluation results...
DONE (t=0.06s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.600
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.971
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.784
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.605
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.663
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.663
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.668
12/17 09:43:44 - mmengine - INFO - bbox_mAP_copypaste: 0.600 0.971 0.784 -1.000 0.000 0.605
12/17 09:43:44 - mmengine - INFO - Epoch(val) [5][6/6]    coco/bbox_mAP: 0.6000  coco/bbox_mAP_50: 0.9710  coco/bbox_mAP_75: 0.7840  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: 0.0000  coco/bbox_mAP_l: 0.6050  data_time: 0.0342  time: 0.2558
```

:::


QAT 训练完毕后，会自动导出量化后的模型，您可以使用以下命令对其进行验证：

```sh
python3 tools/test.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    work_dirs/rtmdet_nano_8xb32_300e_coco/qat/qat_model_int8.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    imgsz='(192,192)'
```

评估得到的精度如下：

```sh
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.091
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.209
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.069
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.094
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.138
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.185
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.314
```

