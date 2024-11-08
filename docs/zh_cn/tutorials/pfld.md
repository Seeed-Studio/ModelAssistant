# 关键点回归训练及测试

本节将介绍如何使用PFLD模型，内容包括以下：

1. 数据准备
2. 模型训练
3. 模型导出
4. 测试 


# 数据准备

数据集类似YOLO格式，每张图片有对应的注释信息，其文件架构应如下所示：
```bash
data_root
    |
    |--train
    |    |--annotations.txt
    |    |--images
    |          |--xxx.jpg
    |          |--xxx.jpg
    |
    |----val
    |    |--annotations.txt
    |    |--images
    |          |--xxx.jpg
    |          |--xxx.jpg
```
tips：需特别注意数据集的根目录的路径，在后续训练、导出、测试中均会用到。

# 模型训练

在满足数据集文件结构的情况下用户只需要修改数据集根路径即可进行训练，其训练命令如下：

```bash
python tools/train.py configs/pfld/pfld_mbv2_1000e.py --cfg-options data_root=${数据集路径}
```
训练完成后默认会在work_dirs路径下生成相应的模型权重文件，同时会有以时间命名的文件夹，此文件夹内含有训练日志相关的文件。


# 模型导出

在训练完成后即可进行导出用户需要的格式，其导出命令如下，更多详细信息可通过`python tools/export.py --help`命令查看

```bash
python tools/export.py configs/pfld/pfld_mbv2_1000e.py work_dirs/epoch_100.pth --imgsz ${输入图像宽高} --format ${导出模型格式}
```
导出的模型与模型在相同路径，相同文件名，但是不同后缀。

# 测试

目前测试可运行`Pytorch`格式的模型以及导出的其他格式的模型，其运行命令如下：

```bash
python tools/test.py configs/pfld/pfld_mbv2_1000e.py work_dirs/epoch_100.onnx  --cfg-options data_root=${数据集路径}
```
测试完成后将会显示模型的Accuracy结果，此结果应与训练模型结果对齐。
