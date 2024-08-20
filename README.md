# Project Title

A brief description of what this project does and who it's for.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
<!-- - [Contributing](#contributing)
- [License](#license)
- [Contact](#contact) -->

## Installation

该项目基于主项目的环境进行编写，但仍旧需要安装一些新增的库

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Dataset Manipulation
本项目提供两种数据集的训练方式：```"Dynamic_Train"```与```"Train"```，在```main.py```函数中可以进行指定。

当指定为```"Dynamic_Train"```时，模型将会实时从指定串口中（串口号请在main.py函数中指定）读取相关的数据，目前该方式支持音频数据与三轴振动信号数据。

当指定为```"Train"```时，需要以如下方式创建数据集并读取数据集：
```bash
# 读取指定串口的数据，ctrl+c暂停读取数据并将文件打包为.csv格式
python ./dataset_tool/serial_port_read.py

# 处理.csv格式文件，对每条信号数据进行分割，转换为塔格拉姆角矩阵数据，并以.npy格式保存每个矩阵
# 该程序会在主文件夹下生成一个datasets文件夹，相关.npy文件将会存放在路径./datasets/Train/ 下
python ./dataset_tool/Signal_data_processing.py
```
目前该方式仅支持三轴振动信号数据，且无需手动划分验证集，在main.py中将自动划分训练与验证集

### Train
运行```main.py```即可，训练完毕后将会在主文件夹下生成一个checkpoint文件夹，该文件夹下保存了最好的一版模型，供evaluate.py使用。大部分参数无需更改。

### Evaluate

在```./evaluate.py```中指定串口与需要评估的模型后，运行即可，输出为正/异常状态。需要注意设置异常阈值，异常阈值有两个，一个为静止模态检测阈值，一个为运动模态检测阈值，建议先观察正常情况下两个异常分数的最低点，再进行设置

### Note

尽管在目前参数下的灵敏度与稳定性得到了平衡，但若一些任务有特殊需要，想要更加稳定或更加灵敏的性能，这里建议一个重要参数：```./dataset_tool/tools.py``` 中的sample_rate参数，在确保机器运行稳定，且需要更加灵敏时请加大该参数以便模型获得更大的时间窗口学习整体运动规律，在需要无视机械运动中的正常噪声且仅关心较大的运动姿态变化异常时请减小该参数以便模型仅捕捉瞬时的运动姿态变化。

注意每次更改都需要重复采集数据的过程！


```
demo
├─ checkpoints
│  └─ best-checkpoint-v189.ckpt
├─ ckpt2tflite.py
├─ datasets
│  └─ Train
├─ dataset_tool
│  ├─ serial_port_read.py
│  ├─ Signal_data_processing.py
│  ├─ tools.py
│  └─ __init__.py
├─ environment.yml
├─ evaluate.py
├─ export.py
├─ export_quant.py
├─ main.py
├─ misc
│  ├─ tools.py
│  └─ __init__.py
├─ model
│  ├─ Block
│  │  ├─ ConvBlock.py
│  │  ├─ Decoder.py
│  │  ├─ Encoder.py
│  │  ├─ Soft_cluster.py
│  │  └─ __init__.py
│  ├─ models.py
│  └─ __init__.py
├─ README.md
├─ robot_arm_control
│  ├─ horizontal_drag_custom.py
│  ├─ http_simple_ctrl.py
│  ├─ motion_loop.py
│  ├─ requirements.txt
│  ├─ serial_simple_ctrl.py
│  └─ vertical_drag_custom.py
├─ seed.npy
├─ serial_data.csv
├─ test.py
├─ test2.py
├─ test_int8.tflite
├─ tflite_evaluate.py
├─ tflite_vs_ckpt.py
├─ x_data_mean.npy
└─  x_data_std.npy

```