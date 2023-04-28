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

### Step 0 - 克隆 Git 仓库

首先，您需要将 [EdgeLab 项目源代码](https://github.com/Seeed-Studio/EdgeLab)克隆到本地，我们使用 Git 管理并将其托管在 GitHub，在下方提供了两种不同的克隆方法（任选其一既可）。如果您还未安装 Git，可以参考 [Git 官方文档](https://git-scm.com/book/zh/v2/%E8%B5%B7%E6%AD%A5-%E5%AE%89%E8%A3%85-Git)在您的计算机上配置 Git。

::: code-group

```sh [HTTPS]
git clone https://github.com/Seeed-Studio/EdgeLab.git
# 进入 EdgeLab 项目目录
cd EdgeLab
```

```sh [SSH]
git clone git@github.com:Seeed-Studio/EdgeLab.git
# 进入 EdgeLab 项目目录
cd EdgeLab
```

:::

### Step 1 - 创建虚拟环境

假设您已经安装了 Conda，首先**创建**并**激活**一个 Conda 虚拟环境。

```sh
conda create --name edgelab python=3.8 -y
# 激活 EdgeLab 虚拟环境
conda activate edgelab
```

### Step 2 - 安装 PyTorch

EdgeLab 依赖 PyTorch，在运行下方代码前，请再次确认你已经已经**激活**了刚刚创建的虚拟环境。

对于带有 GPU (CUDA) 的设备，我们建议安装支持 GPU 加速的依赖项。我们列出了您在两种不同情况下可以选择的配置选项，请根据您的硬件环境手动选择。

- 仅限 CPU 平台:

    ::: code-group

    ```sh [conda]
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```

    ```sh [pip]
    pip3 install torch torchvision torchaudio
    ```

    :::

- 包含 GPUs (CUDA) 的平台:

    ::: code-group

    ```sh [conda]
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia
    ```

    ```sh [pip]
    # 如果您不在虚拟环境中，请谨慎选择 CUDA 版本。例如，这里我们使用 CUDA 11.7
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
    ```

    :::

::: tip
如果您的平台没有为 NVIDIA GPU 安装 CUDA，您可以在 [NVIDIA CUDA Toolkit Archive 网站](https://developer.nvidia.com/cuda-toolkit-archive)上找到 CUDA 安装程序。我们建议在您的主机环境中使用 CUDA 11.7 或更高版本。此外，如果要在其他平台上安装 PyTorch，请在 [PyTorch 官方网站](https://pytorch.org/get-started/locally/)上阅读更多信息。
:::

### Step 3 - 安装基本依赖项

**请确认您已经激活了虚拟环境并处在 EdgeLab 的源代码主工作目录**，然后运行以下代码完成基本依赖项的配置。

```sh
# pip 安装 EdgeLab 基础依赖
pip3 install -r requirements/base.txt
# mim 安装 OpenMMLab 依赖
mim install -r requirements/mmlab.txt
# mim 安装 EdgeLab 包
mim install -e .
```

### Step 4 - 安装额外的依赖项 (可选)

如果您需要进行模型转换或者推理测试，您还需要额外安装以下依赖。

```sh
# 安装推理依赖
pip3 install -r requirements/inference.txt
```


## 其他方法

如果你已经配置好了 Conda，EdgeLab 环境的配置可以在 Linux 上使用 Shell 脚本自动完成 (在 Ubuntu 20.04~22.10 上测试)。

```bash
bash scripts/setup_linux.sh
```

或者您也可以使用 Conda 的配置文件手动完成配置。

::: code-group

```sh [CPU]
conda env create -n edgelab -f environment.yml
# 激活 EdgeLab 虚拟环境
conda activate edgelab
# pip 安装全部依赖 (mmcv 需要编译，可能需要一定的时间)
pip3 install -r requirements.txt
# mim 安装 EdgeLab 包
mim install -e .
```

```sh [GPU (CUDA)]
conda env create -n edgelab -f environment_cuda.yml
# 激活 EdgeLab 虚拟环境
conda activate edgelab
# pip 安装全部依赖 (mmcv 需要编译，可能需要一定的时间)
pip3 install -r requirements.txt
# mim 安装 EdgeLab 包
mim install -e .
```

:::


## 注意事项

在完成了 Miniconda 的安装与使用 Conda 配置 EdgeLab 后，我们创建了名为 `edgelab` 的 Conda 虚拟环境，并在虚拟环境中安装了依赖项。在之后与 EdgeLab 相关的配置和开发过程中，请确保您处在 EdgeLab 的虚拟环境中，您可使用以下命令激活 EdgeLab 虚拟环境:

```sh
conda activate edgelab
```

如果您想重新配置或移除 EdgeLab 虚拟环境，您可以运行以下命令:

```sh
conda env remove -n edgelab
```


## FAQs

- 从 Anaconda 的默认通道安装软件包时，连接速度较慢。

    请耐心等待并尝试一些第三方镜像渠道，如 [SJTU mirror](https://mirror.sjtu.edu.cn/docs/anaconda)，[TUNA mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda) 等。
