# Installation

The EdgeLab runtime environment requires [PyTorch](https://pytorch.org/get-started/locally/) and following [OpenMMLab](https://openmmlab.com/) third-party libraries:

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab Computer Vision Foundation Library.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolkit and benchmarking. In addition to classification tasks, it is also used to provide a variety of backbone networks.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab inspection toolbox and benchmark.
- [MIM](https://github.com/open-mmlab/mim): MIM provides a unified interface for starting and installing the OpenMMLab project and its extensions, and managing the OpenMMLab model library.

## Prerequisites

EdgeLab works on Linux, Windows, and macOS. **We strongly recommend you to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage python packages.** Please follow the steps below to prepare the environment.

:::tip
Miniconda is a free minimal installer for conda, you can download and install Miniconda3 from [Miniconda Official Website](https://docs.conda.io/en/latest/miniconda.html).
:::

### Step 0 - Clone the Git Repository

First, you need to clone the [EdgeLab Source Code](https://github.com/Seeed-Studio/EdgeLab) locally. We use Git to manage and host it on GitHub, and provide two different ways to clone it below (choose either one). If you don't have Git installed, you can configure Git on your computer by referring to the [Git Documentation](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

::: code-group

```sh [HTTPS]
git clone https://github.com/Seeed-Studio/EdgeLab.git --depth 1 && \
cd EdgeLab
```

```sh [SSH]
git clone git@github.com:Seeed-Studio/EdgeLab.git --depth 1 && \
cd EdgeLab
```

:::

### Step 1 - Create Virtual Environment

Assuming you have conda installed, then **create** and **activate** a conda virtual environment.

```sh
conda create --name edgelab python=3.8 -y && \
conda activate edgelab
```

### Step 2 - Install PyTorch

EdgeLab relies on PyTorch. Before running the following code, please confirm again that you have **activated** the virtual environment you just created.

For devices with GPUs (CUDA), we recommend installing dependencies that support GPU acceleration. We have listed the configuration options you can choose from in 2 different cases, please choose manually according to your hardware environment.

- CPU-Only platform:

  ::: code-group

  ```sh [conda]
  conda install pytorch torchvision torchaudio cpuonly -c pytorch
  ```

  ```sh [pip]
  pip3 install torch torchvision torchaudio
  ```

  :::

- GPUs (CUDA) platform:

  ::: code-group

  ```sh [conda]
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia
  ```

  ```sh [pip]
  pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
  ```

  :::

::: tip
You can find CUDA installers on [NVIDIA CUDA Toolkit Archive Website](https://developer.nvidia.com/cuda-toolkit-archive) if your platform have not CUDA installed for NVIDIA GPUs, we recommend to use CUDA 11.7 or above on your host environment. For installing PyTorch on other platforms, please read more on [PyTorch Official Website](https://pytorch.org/get-started/locally/).
:::

### Step 3 - Install Essential Dependencies

**Please confirm that you have activated the virtual environment and in the main working directory of EdgeLab source code**, and then run the following code to complete the configuration of the basic dependencies.

- Install EdgeLab deps

```sh
pip3 install -r requirements/base.txt && \
mim install -r requirements/mmlab.txt && \
mim install -e .
```

### Step 4 - Install Extra Dependencies (Optional)

If you need to perform model transformation or inference testing, you also need to install the following additional dependencies.

```sh
pip3 install -r requirements/inference.txt -r requirements/export.txt
```

If you wish to make changes to EdgeLab and submit them to us, we recommend that you additionally run the following command to facilitate checking your code at commit time.

```sh
pip3 install -r requirements/tests.txt
pre-commit install
```

## Other Method

The configuration of EdgeLab environment can be done automatically using a shell script on Linux (tested on Ubuntu 20.04~22.10), if you have Conda setup.

```bash
bash scripts/setup_linux.sh
```

Or you can do the configuration manually using Conda's configuration file.

::: code-group

```sh [CPU]
conda env create -n edgelab -f environment.yml -y && \
conda activate edgelab && \
pip3 install -r requirements.txt && \
mim install -e .
```

```sh [GPU (CUDA)]
conda env create -n edgelab -f environment_cuda.yml -y && \
conda activate edgelab && \
pip3 install -r requirements_cuda.txt && \
mim install -e .
```

:::

## Reminders

After completing the installation of Miniconda and configuring EdgeLab with Conda, we created a Conda virtual environment named `edgelab` and installed the dependencies in the virtual environment. For subsequent EdgeLab-related configuration and development, make sure you are in the EdgeLab virtual environment, which you can activate with the following command.

```sh
conda activate edgelab
```

If you want to reconfigure or remove the EdgeLab virtual environment, you can run the following command.

```sh
conda env remove -n edgelab
```

## FAQs

- I have slow connection speed while installing packages from anaconda's default channels.

  Please be patient and try some third-party mirrored channels, such as [SJTU mirror](https://mirror.sjtu.edu.cn/docs/anaconda), [TUNA mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda) and etc.
