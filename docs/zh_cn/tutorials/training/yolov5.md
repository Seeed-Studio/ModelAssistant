# Yolo模型训练

本节描述了如何在COCO数字电表数据集上训练数字电表模型。Yolo数字电表检测模型的实现是基于[yolov5](https://github.com/ultralytics/yolov5)和[mmyolo](https://github.com/open-mmlab/mmyolo)的动力。



## 准备数据集

本教程使用[数字数据集](https://universe.roboflow.com/seeeddatasets/seeed_meter_digit/)来训练yolov5模型，请参考以下步骤来完成数据集的准备。

1. 用COCO数据集模式下载数字仪表数据集

2. 记住解压后的数据集的**文件夹路径（如`datasets/digital_meter`），以后可能需要使用这个文件夹路径。


## 选择一个配置

我们将根据我们需要执行的训练任务的类型选择一个合适的配置文件，我们已经在[Config](../config.md)中介绍了配置文件的功能、结构和原理。

对于yolov5模型的例子，我们使用`yolov5tiny.py`作为配置文件，它位于EdgeLab根目录`configs/yolov5`下的文件夹中，其另外继承了`base_arch.py`配置文件。

对于初学者，我们建议首先注意这个配置文件中的`data_root`和`epochs`参数。

::详情 `yolov5tiny.py`.

```python
_base_='./_base_/default_runtime_det.py'
_base_ = ["./base_arch.py"]

锚点 = [...
    [(10, 13), (16, 30), (33, 23)], # P3/8
    [(30, 61), (62, 45), (59, 119)], # P4/16
    [(116, 90), (156, 198), (373, 326)] # P5/32
]
num_classes = 11
deepen_factor = 0.33
加宽因子 = 0.15

strides = [8, 16, 32］

model = dict(
    type='mmyolo.YOLODetector'、
    backbone=dict(
        type='YOLOv5CSPDarknet'、
        deepen_factor=deepen_factor、
        widen_factor=widen_factor、
    ),
    颈部=dict(
        type='YOLOv5PAFPN'、
        deepen_factor=deepen_factor、
        widen_factor=widen_factor、
    ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes、
            in_channels=[256, 512, 1024]、
            widen_factor=widen_factor、
        ),
    ),
)
```

:::


## 训练模型

训练模型需要使用我们之前配置的EdgeLab工作环境，如果你按照我们的[安装](../../introduction/installation.md)指南使用Conda将EdgeLab安装在一个名为`edgelab`的虚拟环境中，请首先确保你当前处于虚拟环境。

然后，在EdgeLab项目根目录下，我们执行以下命令来训练一个yolov5数字仪表检测模型。

```sh
python3 tools/train.py ``sh
    det ```det ```
    configs/yolov5/yolov5tiny.py
    --cfg-options / --cfg-options
        
                                epochs=50
```

在训练过程中，模型权重和相关日志信息默认保存在`work_dirs/yolov5tiny`路径下，你可以使用[TensorBoard](https://www.tensorflow.org/tensorboard/get_started)等工具来监测训练情况。

``sh
tensorboard --logdir work_dirs/word_dirs/yolov5tiny
```

训练完成后，最新的yolov5模型权重文件的路径被保存在`work_dirs/yolov5tiny/last_checkpoint`文件中。请注意权重文件的路径，因为将模型转换为其他格式时需要它。

提示

如果你已经配置了一个虚拟环境但没有激活，你可以用以下命令激活它。

```sh
conda activate edgelab
```

:::

### 参数说明

关于模型训练期间的更多参数，你可以参考下面的代码。

::: 代码组

```sh [模型类型]
python3 tools/train.py ````。
    det `// [！代码焦点]
    configs/yolov5/yolov5tiny.py ``s
    --work-dir work_dir \\
    --gpu-id 0
    --cfg-options
        data_root='datasets/digital_meter' \\
        epochs=50
```

``sh [配置路径]
python3 tools/train.py ``````。
    
    configs/yolov5/yolov5tiny // [!］
    --work-dir work_dir
    --gpu-id 0
    --cfg-options
        data_root='datasets/digital_meter' （数据集/数字表）。
        epochs=50
```

``sh [工作目录]
python3 tools/train.py
    
    configs/yolov5/yolov5tiny.py
    --work-dir work_dir // [!code focus] 。
    --gpu-id 0
    --cfg-options
        data_root='datasets/digital_meter' \\
        epochs=50
```

``sh [GPU ID]
python3 tools/train.py
    
    configs/yolov5/yolov5tiny.py
    
    --gpu-id 0 // [!］
    --cfg-options
        data_root='datasets/digital_meter' \
        epochs=50
```

```sh [配置覆盖]
python3 tools/train.py ````。
    det `s
    configs/yolov5/yolov5tiny.py （配置覆盖
    
    --gpu-id 0
    --cfg-options // [!code focus] [!
        data_root='datasets/digital_meter' // [!］
        epochs=50 // [!］
```

:::


## 测试和评估

### 测试

在完成了yolov5模型的训练后，你可以指定特定的权重，并使用以下命令测试该模型。

```sh
python3 tools/test.py ``sh
    
    configs/yolov5/yolov5tiny.py ``sh python3 tools/test.py
    "$(cat work_dirs/yolov5tiny/last_checkpoint)" \
    --cfg-options
        data_root='datasets/digital_meter'。
```

::提示

如果你想在测试时进行实时预览，你可以在测试命令中附加一个参数`--show`来显示预测的结果。更多的可选参数，请参考源代码`tools/test.py`。

:::

### 评估

为了在现实的边缘计算设备上进一步测试和评估该模型，你需要导出该模型。在导出模型的过程中，EdgeLab会对模型进行一些优化，如模型修剪、提炼等。您可以参考[Export](../export/overview)部分，了解更多的信息。

### 部署

在导出模型后，你可以将模型部署到边缘计算设备上进行测试和评估。你可以参考 [examples](../../examples/examples.md) 部分来了解更多关于如何部署模型的信息。