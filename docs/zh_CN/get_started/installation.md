# 环境安装
- [环境安装](#环境安装)
    - [先决条件](#先决条件)
    - [其他方式](#其他方式)
    - [提醒](#提醒)
    - [FAQs](#faqs)

EdgeLab的运行环境需要[PyTorch](https://pytorch.org/get-started/locally/)和以下[OpenMMLab](https://openmmlab.com/)第三方库。

- [MMCV](https://github.com/open-mmlab/mmcv)：OpenMMLab计算机视觉基础库。
- [MMClassification](https://github.com/open-mmlab/mmclassification)：OpenMMLab图像分类工具包和基准测试。除了分类任务外，它还被用来提供各种骨干网络。
- [MMDetection](https://github.com/open-mmlab/mmdetection)：OpenMMLab检测工具箱和基准测试。
- [MMDPose](https://github.com/open-mmlab/mmpose)：OpenMMLab检测工具箱和基准测试。
- [MIM](https://github.com/open-mmlab/mim)：MIM为启动和安装 OpenMMLab 项目及其扩展以及管理 OpenMMLab 模型库提供了一个统一的接口。

## 先决条件
**我们强烈建议使用Anaconda3来管理python软件包。** 你可以在完成第一个步骤后使用[script](#other-method)去配置环境，也可以按照下面的步骤配置环境。

**Step 0.** 参照[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载可安装Miniconda。

**Step 1.** 创建conda环境并激活它。

```bash
conda create --name edgelab python=3.8 -y
# activate edgelab
conda activate edgelab
```

**Step 2.** 分别安装GPU支持和CPU支持的包，这取决于你的设备。

GPU平台：

- 请参照[官方文档](https://developer.nvidia.com/cuda-downloads)安装cuda.

- 安装pytorch
    ```bash
    # conda 安装
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

    # pip 安装
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    ```

CPU平台：

- 安装pytorch
    ```bash
    # conda安装
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

    # pip安装
    pip3 install torch torchvision torchaudio
    ```

**Step 3.** 安装依赖库

```bash
# pip安装，conda无法完全安装
pip3 install -r requirements/base.txt
```

**Step 4.** 使用MIM安装MMCV

```bash
pip3 install -U openmim
# 必须通过mim安装
mim install mmcv-full==1.7.0 
```


## 其他方式
项目环境的配置可以在ubuntu 20.04上用一个脚本自动完成，如果你使用的是其他系统，则可以手动完成。所有相关的环境都可以在ubuntu上用以下命令来配置。

```bash
python3 tools/env_config.py
```
**注意:** 上述环境配置时间可能因网络环境不同而不同。


## 提醒

在上诉步骤完成后，所需的环境变量已经被添加到~/.bashrc文件中。一个名为edgelab的conda虚拟环境已经被创建，并且在虚拟环境中安装了依赖项。如果它还没有被激活，你可以用以下命令激活conda、虚拟环境和其他相关的环境变量。

```bash
source ~/.bashrc
conda activate edgelab
```

## FAQs
- 