# 安装

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 运行环境依赖于 PyTorch 和来自 OpenMMLab 的各种第三方库。您可以在 [GitHub](https://github.com/Seeed-Studio/ModelAssistant) 上找到 SSCMA 的代码。

## 准备工作

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 可在 Linux、Windows 和 macOS 上运行。我们强烈建议您使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) 来管理 Python 包。请按照以下步骤准备环境。

:::tip
Miniconda 是 conda 的免费最小安装程序，您可以从 [Miniconda 官方网站](https://docs.conda.io/en/latest/miniconda.html)下载并安装 Miniconda3。Mamba 是一个更快的 conda 包管理器，可以替代 conda，您可以从 [Mamba 官方网站](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)下载并安装 Mamba。
:::

### 第 1 步 - 获取源代码

首先，您需要将 [SSCMA 源代码](https://github.com/Seeed-Studio/ModelAssistant)下载或克隆到本地。对于后者，我们使用 Git 将它托管它在 GitHub 上，这里提供了以下两种不同的克隆方式（选择其中一种）。如果您没有安装 Git，可以参考 [Git 文档](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 在您的计算机上配置 Git。

:::code-group

```sh [HTTPS]
git clone https://github.com/Seeed-Studio/ModelAssistant.git --depth 1 && \
cd ModelAssistant
```

```sh [SSH]
git clone git@github.com:Seeed-Studio/ModelAssistant.git --depth 1 && \
cd ModelAssistant
```

:::

### 第 2 步 - 创建虚拟环境

假设您已经安装了虚拟环境，然后我们**创建**和**激活**一个新的虚拟环境。

:::code-group

```sh [Conda]
conda create --name sscma python=3.12 -y && \
conda activate sscma
```

```sh [Mamba]
mamba create --name sscma python=3.12 -y && \
mamba activate sscma
```

```sh [Micromamba]
micromamba create --name sscma python=3.12 -y && \
micromamba activate sscma
```

:::

:::tip
请注意，我们使用 Python 3.12 作为虚拟环境的默认版本，您可以根据需要更改版本号。
:::

### 第 3 步 - 安装基本依赖项

**请确认您已激活虚拟环境，并位于 [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 源代码的主工作目录中**，然后运行以下代码来完成基本依赖项的安装。

:::code-group

```sh [pip]
python3 -m pip install -r requirements.txt
```

```sh [uv]
python3 -m pip install uv && \
uv install -r requirements.txt
```

:::

至此，您已经成功安装了 SSCMA，您可以运行以下命令来验证是否安装成功。

```sh
python3 tools/train.py --help
```

:::details 点击查看输出示例

```sh
usage: train.py [-h] [--amp] [--auto-scale-lr] [--resume] [--work_dir WORK_DIR] [--cfg-options CFG_OPTIONS [CFG_OPTIONS ...]]
                [--launcher {none,pytorch,slurm,mpi}] [--local_rank LOCAL_RANK]
                config

Train a detector

positional arguments:
  config                train config file path

options:
  -h, --help            show this help message and exit
  --amp                 enable automatic-mixed-precision training
  --auto-scale-lr       enable automatically scaling LR.
  --resume              resume from the latest checkpoint in the work_dir automatically
  --work_dir WORK_DIR, --work-dir WORK_DIR
                        the dir to save logs and models
  --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
                        override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If
                        the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple
                        values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.
  --launcher {none,pytorch,slurm,mpi}
                        job launcher
  --local_rank LOCAL_RANK, --local-rank LOCAL_RANK
```

:::

:::tip
如果您的平台尚未安装 NVIDIA GPU 的 CUDA，请在[NVIDIA CUDA 工具包存档网站](https://developer.nvidia.com/cuda-toolkit-archive)上找到 CUDA 安装程序，我们建议在主机环境中使用 CUDA 11.8 或更高版本。有关在其他平台上安装 PyTorch 的方法，请阅读 [PyTorch 官方网站](https://pytorch.org/get-started/locally/) 的更多信息。
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

- 从 Anaconda 的默认 Channel 安装软件包时，我连接速度很慢。

  请耐心等待，并尝试使用某些第三方镜像源，例如 [SJTU 镜像网站](https://mirror.sjtu.edu.cn/docs/anaconda)、[TUNA 镜像网站](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda)等。

- 从 Pypi 安装软件包时，我的连接速度很慢。

  请耐心等待，并尝试使用某些第三方镜像源，例如[清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/help/pypi)、[上海交通大学镜像站](https://mirror.sjtu.edu.cn/docs/pypi)等。
