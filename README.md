# Backbone example on OpenMMLab framework

English | [简体中文](/README_zh-CN.md)

## Introduction

This is an template repo about how to use OpenMMLab framework to develop a new backbone for multiple vision tasks.

With OpenMMLab framework, you can easily develop a new backbone and use MMClassification, MMDetection and MMSegmentation to benchmark your backbone on classification, detection and segmentation tasks.

## Setup environment

It requires [PyTorch](https://pytorch.org/get-started/locally/) and the following OpenMMLab packages:

- [MIM](https://github.com/open-mmlab/mim): A command-line tool to manage OpenMMLab packages and experiments.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark. Besides classification, it's also a repository to store various backbones.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.

Assume you have prepared your Python and PyTorch environment, just use the following command to setup the environment.

```bash
pip install openmim mmcls mmdet mmsegmentation
mim install mmcv-full
```

## Data preparation

The data structure looks like below:

```text
data/
├── imagenet
│   ├── train
│   ├── val
│   └── meta
│       ├── train.txt
│       └── val.txt
├── ade
│   └── ADEChallengeData2016
│       ├── annotations
│       └── images
└── coco
    ├── annotations
    │   ├── instance_train2017.json
    │   └── instance_val2017.json
    ├── train2017
    └── val2017
```

Here, we only list the minimal files for training and validation on ImageNet (classification), ADE20K (segmentation) and COCO (object detection).

If you want benchmark on more datasets or tasks, for example, panoptic segmentation with MMDetection,
just organize your dataset according to MMDetection's requirements. For semantic segmentation task,
you can organize your dataset according to this [tutorial](https://mmsegmentation.readthedocs.io/en/latest/dataset_prepare.html)

## Usage

### Implement your backbone

In this example repository, we use the ConvNeXt as an example to show how to implement a backbone quickly.

1. Create your backbone file and put it in the `models` folder. In this example, [`models/convnext.py`](models/convnext.py).

   In this file, just implement your backbone with PyTorch with two modifications:

   1. The backbone and modules should inherits `mmcv.runner.BaseModule`. The
      `BaseModule` is almost the same as the `torch.nn.Module`, and supports using
      `init_cfg` to specify the initizalization method includes pre-trained model.

   2. Use one-line decorator as below to register the backbone class to the `mmcls.models.BACKBONES` registry.
      ```python
      @BACKBONES.register_module(force=True)
      ```
      > :question: What is registry? Have a look at [here](https://mmcv.readthedocs.io/en/latest/understand_mmcv/registry.html)!

2. **[Optional]** If you want to add some extra components for specific task, you
   can also add it refers to [`models/det/layer_decay_optimizer_constructor.py`](models/det/layer_decay_optimizer_constructor.py).

3. Add your backbone class and custom components to [`models/__init__.py`](models/__init__.py).

### Create config files

Add your config files for each task to [`configs/`](./configs). If your are not familiar with config files,
the [tutorial](https://mmclassification.readthedocs.io/en/latest/tutorials/config.html#config-file-structure) can help you.

In a word, use base config files of model, dataset, schedule and runtime to
compose your config files. Of course, you can also override some settings of
base config in your config files, even write all settings in one file.

In this template, we provide a suit of popular base config files, you can also
find more useful base configs from [mmcls](https://github.com/open-mmlab/mmclassification/tree/master/configs/_base_),
[mmdet](https://github.com/open-mmlab/mmdetection/tree/master/configs/_base_) and
[mmseg](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/_base_).

### Training and testing

For training and testing, you can directly use mim to train and test the model

At first, you need to add the current folder the the `PYTHONPATH`, so that Python can find your model files.

```shell
export PYTHONPATH=`pwd`:$PYTHONPATH 
```

#### On local single GPU:

```bash
# train classification models
mim train mmcls $CONFIG --work-dir $WORK_DIR

# test classification models
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)"

# train object detection / instance segmentation models
mim train mmdet $CONFIG --work-dir $WORK_DIR

# test object detection / instance segmentation models
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm

# train semantic segmentation models
mim train mmseg $CONFIG --work-dir $WORK_DIR

# test semantic segmentation models
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU
```

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- CHECKPOINT: the path of the checkpoint downloaded from our model zoo or trained by yourself

#### On multiple GPUs (4 GPUs here):

```bash
# train classification models
mim train mmcls $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4

# test classification models
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)" --launcher pytorch --gpus 4

# train object detection / instance segmentation models
mim train mmdet $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4

# test object detection / instance segmentation models
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm --launcher pytorch --gpus 4

# train semantic segmentation models
mim train mmseg $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4 

# test semantic segmentation models
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU --launcher pytorch --gpus 4
```

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- CHECKPOINT: the path of the checkpoint downloaded from our model zoo or trained by yourself

#### On multiple GPUs in multiple nodes with Slurm (total 16 GPUs here):

```bash
# train classification models
mim train mmcls $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# test classification models
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)" --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# train object detection / instance segmentation models
mim train mmdet $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# test object detection / instance segmentation models
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# train semantic segmentation models
mim train mmseg $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# test semantic segmentation models
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

- CONFIG: the config files under the directory `configs/`
- WORK_DIR: the working directory to save configs, logs, and checkpoints
- CHECKPOINT: the path of the checkpoint downloaded from our model zoo or trained by yourself
- PARTITION: the slurm partition you are using
