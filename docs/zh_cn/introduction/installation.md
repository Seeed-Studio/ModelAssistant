# 安装指南

EdgeLab 运行环境需要 [PyTorch](https://pytorch.org/get-started/locally/) 以及下面的 [OpenMMLab](https://openmmlab.com/) 第三方库:

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库。
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具包和基准测试。除了分类任务外，它还用于提供各种骨干网络。
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab检测工具箱和基准。
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 检查工具箱和基准。
- [MIM](https://github.com/open-mmlab/mim): MIM 提供了一个统一的接口，用于启动和安装 OpenMMLab 项目及其扩展，以及管理 OpenMMLab 模型库。


## 先决条件

EdgeLab 适用于 Linux、Windows 和 macOS。**我们强烈建议您使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 管理 Python 包。** 请按照以下步骤准备环境。

::: tip
Miniconda 是一个免费的 Conda 最小安装程序，您可以从 [Miniconda 官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda3。
:::

### Step 1 - 创建虚拟环境

假设您已经安装了 Conda，首先创建并激活一个 Conda 虚拟环境。

```bash
conda-create--name edgelab python=3.8 -y
# 激活 edgelab 虚拟环境
conda edgelab
```

### Step 2 - 安装 PyTorch

EdgeLab 依赖 PyTorch。对于带有 GPU (CUDA) 的设备，我们建议安装支持 GPU 加速的依赖项。我们列出了您在两种不同情况下可以选择的配置选项，请根据您的硬件环境手动选择。

- 仅限 CPU 平台:

    ::: code-group

    ```bash [conda]
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```

    ```bash [pip]
    pip3 install torch torchvision torchaudio
    ```

    :::

- 包含 GPUs (CUDA) 的平台:

    ::: code-group

    ```bash [conda]
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia
    ```

    ```bash [pip]
    # 如果您不在虚拟环境中，请谨慎选择 CUDA 版本。例如，这里我们使用 CUDA 11.7
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
    ```

    :::

::: tip
如果您的平台没有为 NVIDIA GPU 安装 CUDA，您可以在 [NVIDIA CUDA Toolkit Archive 网站](https://developer.nvidia.com/cuda-toolkit-archive)上找到 CUDA 安装程序。我们建议在您的主机环境中使用 CUDA 11.7 或更高版本。此外如果要在其他平台上安装 PyTorch，请在 [PyTorch 官方网站](https://pytorch.org/get-started/locally/)上阅读更多信息。
:::

### Step 3 - 安装基本依赖项

```bash
# pip 安装 edgelab 依赖
pip3 install -r requirements/base.txt
# mim 安装 mmlab 依赖和 edgelab
mim install -r requirements/mmlab.txt
mim install -e .
```

### Step 4 - 安装额外的依赖项 (可选)

```bash
# 安装推理依赖
pip3 install -r requirements/inference.txt
```


## 其他方法

如果你已经配置好了 Conda，EdgeLab 环境的配置可以在 Linux 上使用 Shell 脚本自动完成 (在 Ubuntu 20.04~22.10 上测试)。

```bash
bash scripts/setup_linux.sh
```


## 提醒

上述步骤完成后，所需的环境变量已添加到 `~/.bashrc` 文件中并创建了名为 `edgelab` 的 Conda 虚拟环境，在虚拟环境中安装了依赖项。如果此时未激活。您可以使用以下命令激活 Conda 虚拟环境并导入其他相关环境变量。

```bash
source ~/.bashrc
conda activate edgelab
```


## FAQs

- 从 Anaconda 的默认通道安装软件包时，连接速度较慢。

    请耐心等待并尝试一些第三方镜像渠道，如 [SJTU mirror](https://mirror.sjtu.edu.cn/docs/anaconda)，[TUNA mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda) 等。
