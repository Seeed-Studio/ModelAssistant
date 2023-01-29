# 安装

EdgeLab的运行环境需要[PyTorch](https://pytorch.org/get-started/locally/)和以下[OpenMMLab](https://openmmlab.com/)第三方库。

- [MMCV](https://github.com/open-mmlab/mmcv)。OpenMMLab计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification)。OpenMMLab图像分类工具包和基准测试。除了分类任务外，它还被用来提供各种骨干网络
- [MMDetection](https://github.com/open-mmlab/mmdetection)。OpenMMLab检测工具箱和基准测试
- [MMDPose](https://github.com/open-mmlab/mmpose): OpenMMLab检测工具箱和基准测试

```{note}
我们强烈建议使用Anaconda3来管理python软件包
```

## 支持 CUDA
### 安装 CUDA
请参考[官方文档](https://developer.nvidia.com/cuda-downloads)
### 安装 pytorch
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
### 安装 MMCV
```bash
pip3 install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```
### 安装 EdgeLab
```bash
pip3 install -r requirements/requirements.txt
```

## 不支持 CUDA
### 安装 pytorch
```bash
pip3 install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0
```
### 安装 MMCY
```bash
pip3 install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html
```
### 安装 EdgeLab
```bash
pip3 install -r requirements/requirements.txt
```

项目环境的配置可以在ubuntu 20.04上用一个脚本自动完成，如果你使用的是其他系统，则可以手动完成。所有相关的环境都可以在ubuntu上用以下命令来配置。

```bash
python3 tools/env_config.py
```

```{warning}
上述环境配置时间可能因网络环境不同而不同。
```

在上诉步骤完成后，所需的环境变量已经被添加到~/.bashrc文件中。一个名为edgelab的conda虚拟环境已经被创建，并且在虚拟环境中安装了依赖项，但此时它还没有被激活。你可以用以下命令激活conda、虚拟环境和其他相关的环境变量。

```bash
source ~/.bashrc
conda activate edgelab
```