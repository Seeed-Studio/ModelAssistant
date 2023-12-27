# 安装

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 运行环境依赖于 PyTorch 和来自 OpenMMLab 的各种第三方库。您可以在 [GitHub](https://github.com/Seeed-Studio/ModelAssistant) 上找到 SSCMA 的代码。要开始，请确保按照[此处](https://pytorch.org/get-started/locally/)的说明，在本地安装了 PyTorch，并获取所需的 OpenMMLab 库。

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库。
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具包和基准测试。除了分类任务外，它还用于提供各种主干网络。
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱和基准测试。
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱和基准测试。
- [MIM](https://github.com/open-mmlab/mim): MIM 提供了一个统一的接口，用于启动和安装 OpenMMLab 项目及其扩展，并管理 OpenMMLab 模型库。

## 准备工作

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 可在 Linux、Windows 和 macOS 上运行。\*\*我们强烈建议您使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 来管理 Python 包。\*\*请按照以下步骤准备环境。

:::tip
Miniconda 是 conda 的免费最小安装程序，您可以从[Miniconda 官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda3。
:::

### 第 0 步 - 克隆 Git 仓库

首先，您需要将 [SSCMA 源代码](https://github.com/Seeed-Studio/ModelAssistant) 克隆到本地。我们使用 Git 来管理和托管它在 GitHub 上，并提供了以下两种不同的克隆方式（选择其中一种）。如果您没有安装 Git，可以参考 [Git 文档](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 在您的计算机上配置 Git。

::: code-group

```sh [HTTPS]
git clone https://github.com/Seeed-Studio/ModelAssistant.git --depth 1 && \
cd SSCMA
```

```sh [SSH]
git clone git@github.com:Seeed-Studio/SSCMA.git --depth 1 && \
cd SSCMA
```

:::

### 第 1 步 - 创建虚拟环境

假设您已经安装了 conda，则**创建**和**激活**一个 conda 虚拟环境。

```sh
conda create --name sscma python=3.8 -y && \
conda activate sscma
```

### 第 2 步 - 安装 PyTorch

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 依赖于 PyTorch。在运行以下代码之前，请再次确认您已经**激活**了刚创建的虚拟环境。

对于带有 GPU（CUDA）的设备，我们建议安装支持 GPU 加速的依赖项。我们列出了您可以根据硬件环境手动选择的配置选项，请根据以下两种情况之一进行选择。

- 仅 CPU 平台：

  ::: code-group

  ```sh [conda]
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  ```

  ```sh [pip]
  pip3 install torch torchvision torchaudio
  ```

  :::

- GPU（CUDA）平台：

  ::: code-group

  ```sh [conda]
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia
  ```

  ```sh [pip]
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
  ```

  :::

::: tip
如果您的平台尚未安装 NVIDIA GPU 的 CUDA，请在[NVIDIA CUDA 工具包存档网站](https://developer.nvidia.com/cuda-toolkit-archive)上找到 CUDA 安装程序，我们建议在主机环境中使用 CUDA 11.7 或更高版本。有关在其他平台上安装 PyTorch 的方法，请阅读 [PyTorch 官方网站](https://pytorch.org/get-started/locally/) 的更多信息。
:::

### 第 3 步 - 安装基本依赖项

**请确认您已激活虚拟环境，并位于 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 源代码的主工作目录中**，然后运行以下代码来完成基本依赖项的配置。

- 安装 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 依赖项

```sh
pip3 install -r requirements/base.txt && \
mim install -r requirements/mmlab.txt && \
mim install -e .
```

### 第 4 步 - 安装额外依赖项（可选）

如果您需要执行模型转换或推理测试，还需要安装以下额外的依赖项。

```sh
pip3 install -r requirements/inference.txt -r requirements/export.txt
```

如果希望对 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 进行更改并将其提交给我们，建议您额外运行以下命令，以便在提交代码时方便检查您的代码。

```sh
pip3 install -r requirements/tests.txt
pre-commit install
```

## 其他方法

可以使用 Linux 上的 shell 脚本自动完成 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 环境的配置（在 Ubuntu 20.04~22.10 上进行了测试），如果您已经设置了 Conda。

```bash
bash scripts/setup_linux.sh
```

或者您可以手动使用 Conda 的配置文件进行配置。

::: code-group

```sh [仅 CPU]
conda env create -n sscma -f environment.yml && \
conda activate sscma && \
pip3 install -r requirements/inference.txt -r requirements/export.txt -r requirements/tests.txt && \
mim install -r requirements/mmlab.txt && \
mim install -e .
```

```sh [GPU（CUDA）]
conda env create -n sscma -f environment_cuda.yml && \
conda activate sscma && \
pip3 install -r requirements/inference.txt -r requirements/export.txt -r requirements/tests.txt && \
mim install -r requirements/mmlab.txt && \
mim install -e .
```

:::

## 提醒事项

完成 Miniconda 的安装和使用 Conda 配置 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 后，我们创建了一个名为 `sscma` 的 Conda 虚拟环境，并在虚拟环境中安装了依赖项。对于后续与 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 相关的配置和开发，请确保您在 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 虚拟环境中，您可以使用以下命令激活它。

```sh
conda activate sscma
```

如果要重新配置或删除 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 虚拟环境，可以运行以下命令。

```sh
conda env remove -n sscma
```

## 常见问题

- 在从 anaconda 的默认 channel 安装软件包时，我连接速度很慢。

  请耐心等待，并尝试使用某些第三方镜像源，例如[SJTU 镜像网站](https://mirror.sjtu.edu.cn/docs/anaconda)、[TUNA 镜像网站](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda)等。
