# 环境安装

- [环境安装](#环境安装)
    - [先决条件](#先决条件)
    - [其他方式](#其他方式)
    - [提醒](#提醒)
    - [FAQs](#faqs)

EdgeLab 的运行环境需要 [PyTorch](https://pytorch.org/get-started/locally/) 和以下 [OpenMMLab](https://openmmlab.com/) 第三方库。

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库。
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具包和基准测试。除了分类任务外，它还被用来提供各种骨干网络。
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱和基准测试。
- [MMDPose](https://github.com/open-mmlab/mmpose): OpenMMLab 检测工具箱和基准测试。
- [MIM](https://github.com/open-mmlab/mim): MIM 为启动和安装 OpenMMLab 项目及其扩展以及管理 OpenMMLab 模型库提供了一个统一的接口。


## 先决条件

**我们强烈建议使用 Anaconda3 来管理 Python 软件包。** 你可以在完成第一个步骤后使用 [Script](#other-method) 去配置环境，也可以按照下面的步骤配置环境。

**Step 0.** 参照[官方网站](https://docs.conda.io/en/latest/miniconda.html)下载可安装 Miniconda。

**Step 1.** 创建 Conda 环境并激活它。

```bash
conda create --name edgelab python=3.8 -y
# 激活 edgelab 虚拟环境
conda activate edgelab
```

**Step 2.** 分别安装 GPU (CUDA) 支持或 CPU 支持的包，这取决于你设备的硬件配制。

GPU 平台: 

- 请参照[官方文档](https://developer.nvidia.com/cuda-downloads)安装 CUDA (11.7 或更新)。

- 安装 PyTorch
    ```bash
    # conda 安装
    conda install cudatoolkit=11.7 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

    # 或: pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
    ```

CPU 平台: 

- 安装 PyTorch
    ```bash
    # conda 安装
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

    # 或: pip3 install torch torchvision torchaudio
    ```

**Step 3.** 安装基础依赖库。

```bash
# pip 安装，conda 无法完全安装
pip3 install -r requirements/base.txt

# mim 安装 mmlab 相关依赖和 edgelab
mim install -r requirements/mmlab.txt
mim install -e .
```

**Step 4 (可选的).** 安装额外的依赖库。

```bash
# audio 依赖
pip3 install -r requirements/audio.txt

# inference 依赖
pip3 install -r requirements/inference.txt

# docs 依赖
pip3 install -r requirements/docs.txt
```


## 其他方式

项目环境的配置可以在 Linux (在 Ubuntu 20.04~22.04 上测试) 上用一个脚本自动完成，如果您使用的是其他操作系统，可以选择手动配置。

```bash
bash scripts/setup.sh
```


## 提醒

在上诉步骤完成后，所需的环境变量已经被添加到 `~/.bashrc` 文件中。一个名为 `edgelab` 的 Conda 虚拟环境已经被创建，并且在虚拟环境中安装了依赖项。如果它还没有被激活，你可以用以下命令激活 Conda 虚拟环境和其他相关的环境变量。

```bash
source ~/.bashrc
conda activate edgelab
```


## FAQs

- 从 Anaconda 的默认通道安装软件包时，连接速度较慢。

    请耐心等待并尝试第三方镜像通道，如 [SJTU 镜像](https://mirror.sjtu.edu.cn/docs/anaconda)，[TUNA 镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)等。
