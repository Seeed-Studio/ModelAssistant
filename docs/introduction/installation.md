# Installation

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) relies on PyTorch and various third-party libraries from OpenMMLab. You can find the SSCMA code on [GitHub](https://github.com/Seeed-Studio/ModelAssistant).

## Prerequisites

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) can run on Linux, Windows, and macOS. We strongly recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to manage Python packages. Please follow these steps to prepare your environment.

:::tip
Miniconda is a free minimal installer for conda, which you can download and install from the [Miniconda official website](https://docs.conda.io/en/latest/miniconda.html). Mamba is a faster conda package manager that can replace conda, and you can download and install Mamba from the [Mamba official website](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).
:::

### Step 1 - Obtain the Source Code

First, you need to download or clone the [SSCMA source code](https://github.com/Seeed-Studio/ModelAssistant) to your local machine. For the latter, we use Git to host it on GitHub, and here are two different cloning methods (choose one). If you do not have Git installed, you can refer to the [Git documentation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to configure Git on your computer.

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

### Step 2 - Create a Virtual Environment

Assuming you have already installed the virtual environment, we then **create** and **activate** a new virtual environment.

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
Please note that we use Python 3.12 as the default version for the virtual environment, and you can change the version number as needed.
:::

### Step 3 - Install Basic Dependencies

**Please ensure that you have activated the virtual environment and are in the main working directory of the [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) source code**, then run the following code to complete the installation of basic dependencies.

:::code-group

```sh [pip]
python3 -m pip install -r requirements.txt
```

```sh [uv]
python3 -m pip install uv && \
uv install -r requirements.txt
```

:::

At this point, you have successfully installed SSCMA. You can run the following command to verify whether the installation was successful.

```sh
python3 tools/train.py --help
```

:::details Click to view the output example

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
If your platform does not have the CUDA for NVIDIA GPU installed, please find the CUDA installer on the [NVIDIA CUDA Toolkit Archive website](https://developer.nvidia.com/cuda-toolkit-archive). We recommend using CUDA 11.8 or higher in your host environment. For more information on installing PyTorch on other platforms, please read the [PyTorch official website](https://pytorch.org/get-started/locally/).
:::

## Reminders

After completing the installation and configuration of Miniconda and setting up [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) with Conda, we have created a Conda virtual environment named `sscma` and installed dependencies in the virtual environment. For subsequent configurations and development related to [SSCMA](https://github.com/Seeed-Studio/ModelAssistant), please ensure that you are in the [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) virtual environment, which you can activate using the following command.

```sh
conda activate sscma
```

If you need to reconfigure or delete the [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) virtual environment, you can run the following command.

```sh
conda env remove -n sscma
```

## Common Issues

- I have slow connection speeds when installing packages from Anaconda's default Channel.

  Please be patient and try using some third-party mirror sources, such as [SJTU Mirror website](https://mirror.sjtu.edu.cn/docs/anaconda), [TUNA Mirror website](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda), etc.

- I have slow connection speeds when installing packages from Pypi.

  Please be patient and try using some third-party mirror sources, such as [Tsinghua University Open Source Software Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi), [Shanghai Jiao Tong University Mirror](https://mirror.sjtu.edu.cn/docs/pypi), etc.
