# Installation
- [Installation](#installation)
    - [GPU support](#gpu-support)
        - [Install cuda](#install-cuda)
        - [Install pytorch](#install-pytorch)
        - [Install dependent libraries](#install-dependent-libraries)
        - [Install MMCV](#install-mmcv)
    - [CPU support](#cpu-support)
        - [Install pytorch](#install-pytorch-1)
        - [Install dependent libraries](#install-dependent-libraries-1)
        - [Install MMCV](#install-mmcv-1)
    - [Other method](#other-method)
    - [Reminder](#reminders)
    - [FAQs](#faqs)

The EdgeLab runtime environment requires [PyTorch](https://pytorch.org/get-started/locally/) and the following [OpenMMLab](https://openmmlab.com/) third-party libraries:

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab Computer Vision Foundation Library
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolkit and benchmarking. In addition to classification tasks, it is also used to provide a variety of backbone networks
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark
- [MMDPose](https://github.com/open-mmlab/mmpose): OpenMMLab inspection toolbox and benchmark
- [MIM](https://github.com/open-mmlab/mim): MIM provides a unified interface for starting and installing the OpenMMLab project and its extensions, and managing the OpenMMLab model library.

**We strongly recommend you to use Anaconda3 to manage python packages.**

## GPU support
1. ### Install cuda
    Please refer to [official install guide](https://developer.nvidia.com/cuda-downloads)

2. ### Install pytorch
    ```bash
    # conda install
    conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

    # pip install
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
    ```

3. ### Install dependent libraries
    ```bash
    # pip install, it is not work for conda.
    pip3 install -r requirements/base.txt
    ```

4. ### Install MMCV
    ```bash
    # must use mim install
    mim install mmcv-full==1.7.0 
    ```

## CPU support
1. ### Install pytorch
    ```bash
    # conda install
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

    # pip install
    pip3 install torch torchvision torchaudio
    ```

2. ### Install dependent libraries
    ```bash
    # pip install, it is not work for conda.
    pip3 install -r requirements/base.txt
    ```

3. ### Install MMCV
    ```bash
    # must use mim install
    mim install mmcv-full==1.7.0 
    ```

## Other method
The configuration of the project environment can be done automatically using a script on ubuntu 20.04, or manually if you are using other systems. All relevant environments can be configured on ubuntu with the following command.

```python
python3 tools/env_config.py
```
**Note:** The above environment configuration time may vary depending on the network environment.


## Reminders

After the appeal steps are completed, the required environment variables have been added to the ~/.bashrc file. A conda virtual environment named edgelab has been created and the dependencies have been installed in the virtual environment, but it is not activated at this point. You can activate conda, the virtual environment and other related environment variables with the following command.

```bash
source ~/.bashrc
conda activate edgelab
```

## FAQs
- 