# Installation

- [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Other Method](#other-method)
    - [Reminders](#reminders)
    - [FAQs](#faqs)

The EdgeLab runtime environment requires [PyTorch](https://pytorch.org/get-started/locally/) and the following [OpenMMLab](https://openmmlab.com/) third-party libraries:

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab Computer Vision Foundation Library
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolkit and benchmarking. In addition to classification tasks, it is also used to provide a variety of backbone networks
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark
- [MMDPose](https://github.com/open-mmlab/mmpose): OpenMMLab inspection toolbox and benchmark
- [MIM](https://github.com/open-mmlab/mim): MIM provides a unified interface for starting and installing the OpenMMLab project and its extensions, and managing the OpenMMLab model library.


## Prerequisites

**We strongly recommend you to use Anaconda3 to manage python packages.** You can use [Script](#other-method) to configure the environment after finishing the **Step 0**, or you can follow all the below steps to prepare the environment.

**Step 0.** Download and install Miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).

**Step 1.** Create a conda environment and activate.

```bash
conda create --name edgelab python=3.8 -y
# activate edgelab env
conda activate edgelab
```

**Step 2.** Install packages for GPU (CUDA) support or CPU support separately, which depends on the hardware configuration of your device.

On GPU (CUDA) platforms:

- Install CUDA (11.7 or later), please refer to [official install guide](https://developer.nvidia.com/cuda-downloads).

- Install PyTorch
    ```bash
    # conda install
    conda install cudatoolkit=11.7 pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

    # or: pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
    ```

On CPU platforms:

- Install PyTorch
    ```bash
    # conda install
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

    # or: pip3 install torch torchvision torchaudio
    ```

**Step 3.** Install essential dependent libraries.

```bash
# pip install, it is not work for conda
pip3 install -r requirements/base.txt

# mim install mmlab deps and edgelab
mim install -r requirements/mmlab.txt
mim install -e .
```

**Step 4 (Optional).** Install extra dependent libraries.

```bash
# audio deps
pip3 install -r requirements/audio.txt

# inference deps
pip3 install -r requirements/inference.txt

# docs deps
pip3 install -r requirements/docs.txt
```


## Other Method

The configuration of EdgeLab environment can be done automatically using a shell script on Linux (tested on Ubuntu 20.04~22.10), or manually config if you are using other operating system.

```bash
bash scripts/setup.sh
```


## Reminders

After the appeal steps are completed, the required environment variables have been added to the `~/.bashrc` file. A conda virtual environment named `edgelab` has been created and the dependencies have been installed in the virtual environment. If it is not activated at this point. You can activate conda, the virtual environment and source other related environment variables with the following command.

```bash
source ~/.bashrc
conda activate edgelab
```


## FAQs

- I have slow connection speed while installing packages from anaconda's default channels.

    Please be patient and try some third-party mirrored channels, such as [SJTU mirror](https://mirror.sjtu.edu.cn/docs/anaconda), [TUNA mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)and etc.
