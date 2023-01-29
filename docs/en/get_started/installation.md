# Installation

The EdgeLab runtime environment requires [PyTorch](https://pytorch.org/get-started/locally/) and the following [OpenMMLab](https://openmmlab.com/) third-party libraries:

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab Computer Vision Foundation Library
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolkit and benchmarking. In addition to classification tasks, it is also used to provide a variety of backbone networks
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmarking
- [MMDPose](https://github.com/open-mmlab/mmpose): OpenMMLab inspection toolbox and benchmarking

```{note}
We strongly recommend to use Anaconda3 to manage python packages
```

## Install with CUDA
### Install cuda
Please refer to [official install guide](https://developer.nvidia.com/cuda-downloads)
### Install pytorch
```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
```
### Install MMCV
```bash
pip3 install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```
### Install EdgeLab
```bash
pip3 install -r requirements/requirements.txt
```

## Install without CUDA
### Install pytorch
```bash
pip3 install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0
```
### Install MMCV
```bash
pip3 install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.10.0/index.html
```
### Install EdgeLab
```bash
pip3 install -r requirements/requirements.txt
```

The configuration of the project environment can be done automatically using a script on ubuntu 20.04, or manually if you are using other systems. All relevant environments can be configured on ubuntu with the following command.

```python
python3 tools/env_config.py
```

```{warning}
The above environment configuration time may vary depending on the network environment.
```

After the appeal steps are completed, the required environment variables have been added to the ~/.bashrc file. A conda virtual environment named edgelab has been created and the dependencies have been installed in the virtual environment, but it is not activated at this point. You can activate conda, the virtual environment and other related environment variables with the following command.

```bash
source ~/.bashrc
conda activate edgelab
```