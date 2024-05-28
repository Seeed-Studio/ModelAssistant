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
目前该方式仅支持三轴振动信号数据

### Train
运行```main.py```即可

### Evaluate

在```./evaluate.py```中指定串口后，运行即可，输出为异常分数。