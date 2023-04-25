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

### Step 1 - Create Virtual Environment

Assuming you have conda installed, then create and activate a conda virtual environment.

```bash
conda create --name edgelab python=3.8 -y
# activate edgelab virtual environment
conda activate edgelab
```

### Step 2 - Install PyTorch

PyTorch is required by EdgeLab. For devices with GPUs (CUDA), we recommend installing dependencies that support GPU acceleration. We have listed the configuration options you can choose from in 2 different cases, please choose manually according to your hardware environment.

- CPU-Only platform:

    ::: code-group

    ```bash [conda]
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```

    ```bash [pip]
    pip3 install torch torchvision torchaudio
    ```

    :::

- GPUs (CUDA) platform:

    ::: code-group

    ```bash [conda]
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 cudatoolkit=11.7 -c pytorch -c nvidia
    ```

    ```bash [pip]
    # please be cautious with CUDA version if you are not in the virtual environment, here for example we use CUDA 11.7
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
    ```

    :::

::: tip
You can find CUDA installers on [NVIDIA CUDA Toolkit Archive Website](https://developer.nvidia.com/cuda-toolkit-archive) if your platform have not CUDA installed for NVIDIA GPUs, we recommend to use CUDA 11.7 or above on your host environment. For installing PyTorch on other platforms, please read more on [PyTorch Official Website](https://pytorch.org/get-started/locally/).
:::

### Step 3 - Install Essential Dependencies

```bash
# pip install edgelab deps
pip3 install -r requirements/base.txt
# mim install mmlab deps and edgelab
mim install -r requirements/mmlab.txt
mim install -e .
```

### Step 4 - Install Extra Dependencies (Optional)

```bash
# install inference deps
pip3 install -r requirements/inference.txt
```


## Other Method

The configuration of EdgeLab environment can be done automatically using a shell script on Linux (tested on Ubuntu 20.04~22.10), if you have Conda setup.

```bash
bash scripts/setup_linux.sh
```


## Reminders

After the appeal steps are completed, the required environment variables have been added to the `~/.bashrc` file. A conda virtual environment named `edgelab` has been created and the dependencies have been installed in the virtual environment. If it is not activated at this point. You can activate conda, the virtual environment and source other related environment variables with the following command.

```bash
source ~/.bashrc
conda activate edgelab
```


## FAQs

- I have slow connection speed while installing packages from anaconda's default channels.

    Please be patient and try some third-party mirrored channels, such as [SJTU mirror](https://mirror.sjtu.edu.cn/docs/anaconda), [TUNA mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda) and etc.
