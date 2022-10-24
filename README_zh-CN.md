# 基于 OpenMMLab 框架的多任务网络模型库

[English](./README.md) | 简体中文

## 简介

这是一个基于 OpenMMLab 框架开发的一个适用于多种视觉任务主干网络的模型训练库。

利用 OpenMMLab 框架，我们可以轻松地开发一个新的主干网络，并利用 MMClassification，MMDetection 和 MMPose 来在分类、检测等任务上进行基准测试。

## 配置环境

运行环境需要 [PyTorch](https://pytorch.org/get-started/locally/) 和以下 OpenMMLab 仓库:

- [MIM](https://github.com/open-mmlab/mim): 一个用于管理 OpenMMLab 包和实验的命令行工具
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱和基准测试。除了分类任务，它同时用于提供多样的主干网络
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱和基准测试
- [MMDPose](https://github.com/open-mmlab/mmpose): OpenMMLab 检测工具箱和基准测试

如果你已经准备好了 Python 和 PyTorch 环境，那么只需要下面两行命令，就可以配置好 mmlab 相关的软件环境。

```bash
pip install openmim mmcls mmdet mmpose          
mim install mmcv-full                       #todo
```

对于上述命令若安装过程太慢可使用国内镜像，例如使用清华镜像可在后面添加`-i https://pypi.tuna.tsinghua.edu.cn/simple` 加快安装速度。
<details>
<summary>Windows</summary>

将本仓库克隆至本地后进入本项目文件夹，同时将本项目文件夹路径添加至环境变量中，变量名为PYTHONPATH，添加完成后可执行以下命令查看是否添加成功。

```bash
set PYTHONPATH
```

若显示本项目地址路径表明添加成功。
</details>

<details>
<summary>Linux</summary>>

同样需要将本项目的文件路径添加至系统环境变量中，变量名为PYTHONPATH，其可通过修改~/.bashrc 文件以保证在后续新终端中可用。
在终端中依次执行以下命令即可：

```bash
echo export PYTHONPATH=`pwd`:\$PYTHONPATH >> ~/.bashrc
source ~/.bashrc
```

</details>

## 数据准备

请将数据集放置在与本项目同级目录下，其数据与本项目结构组织如下所示：\
本仓库示例所使用的数据需要按照如下结构组织：

```text
edgelab/
├── ...
│   .
│   .
datasets/
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

这里，我们只列举了用于训练和验证 ImageNet（分类任务）、ADE20K（分割任务）和 COCO（检测任务）的必要文件。

## 用法

首先我们需要确定所做的任务类型，属于目标检测,分类,或回归，确定后可根据需要选择需要的模型，并选择模型的配置文件。

### 修改配置文件中的数据集路径

这里我们以YOLOv3为例展示如何修改配置文件中的数据集路径

1. 在[configs](./configs)文件夹下寻找所需要修改的[配置文件](./configs/yolo/yolov3_192_node2_person.py)。
2. 在配置文件中找到变量`data_root`，将变量值替换为自己所用数据集的的根目录的路径。
3. 检查配合文件中的`img_perfix`和`ann_file`路径是否正确。
    - `img_perfix` 为数据集图片的路径，`ann_file` 为数据集注释文件的路径

### 训练

1.在上述环境中执行以下命令可开始训练YOLOv3模型。

```shell
mim train mmdet $CONFIG_PATH --gpus=1 --workdir=$WORKERDIR
```

其中`$CONFIG_PATH`需替换为你所使用的模型配置文件本示例中为[yolov3_192_node2_person.py](./configs/yolo/yolov3_192_node2_person.py)的路径。
`$WORKERDIR`为训练过程产生的日志文件和权重文件保存的路径，默认为`worke_dir`

2.在模型训练完成后会在$WORKERDIR文件夹下产生相应的日志文件和模型权重文件(后缀为`.pth`)。

### 训练和测试

对于训练可使用openmim项目下的mim工具，在以上执行`pip install openmim`命令时便已经安装了mim工具，此时可使用如下命令训练相应模型。


<details>
<summary>单机单 GPU：</summary>>

```bash
# 训练分类模型
mim train mmcls $CONFIG --work-dir $WORK_DIR

# 测试分类模型
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)"

# 训练目标检测/实例分割模型
mim train mmdet $CONFIG --work-dir $WORK_DIR

# 测试目标检测/实例分割模型
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm

# 训练语义分割模型
mim train mmseg $CONFIG --work-dir $WORK_DIR

# 测试语义分割模型
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU
```

#### 参数解释

- `$CONFIG`: `configs/` 文件夹下的配置文件路径
- `$WORK_DIR`: 用于保存日志和模型权重文件的文件夹
- `$CHECKPOINT`: 权重文件路径

**[注意]** 添加参数`--gpus=0`可使用CPU进行训练。

</details>
<details>
<summary>单机多 GPU （以 4 GPU 为例）：</summary>

```bash
# 训练分类模型
mim train mmcls $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4

# 测试分类模型
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)" --launcher pytorch --gpus 4

# 训练目标检测/实例分割模型
mim train mmdet $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4

# 测试目标检测/实例分割模型
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm --launcher pytorch --gpus 4

# 训练语义分割模型
mim train mmseg $CONFIG --work-dir $WORK_DIR --launcher pytorch --gpus 4 

# 测试语义分割模型
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU --launcher pytorch --gpus 4
```

#### 参数解释

- `$CONFIG`: `configs/` 文件夹下的配置文件路径
- `$WORK_DIR`: 用于保存日志和模型权重文件的文件夹
- `$CHECKPOINT`: 权重文件路径

</details>
<details>
<summary>多机多 GPU （以 2 节点共计 16 GPU 为例）</summary>

```bash
# 训练分类模型
mim train mmcls $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 测试分类模型
mim test mmcls $CONFIG -C $CHECKPOINT --metrics accuracy --metric-options "topk=(1, 5)" --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 训练目标检测/实例分割模型
mim train mmdet $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 测试目标检测/实例分割模型
mim test mmdet $CONFIG -C $CHECKPOINT --eval bbox segm --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 训练语义分割模型
mim train mmseg $CONFIG --work-dir $WORK_DIR --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION

# 测试语义分割模型
mim test mmseg $CONFIG -C $CHECKPOINT --eval mIoU --launcher slurm --gpus 16 --gpus-per-node 8 --partition $PARTITION
```

#### 参数解释

- `$CONFIG`: `configs/` 文件夹下的配置文件路径
- `$WORK_DIR`: 用于保存日志和模型权重文件的文件夹
- `$CHECKPOINT`: 模型权重文件路径
- `$PARTITION`: 使用的 Slurm 分区

</details>

### 导出ONNX

在模型训练完成后可将pth文件导出到onnx文件格式，并通过onnx转为其他想要使用的格式。
假设此时环境在本项目路径下，可通过执行如下命令导出刚才训练的模型至onnx格式。

```shell
#导出onnx时不对模型进行量化
python ./tools/torch2onnx.py  --model $MODEL_PATH --output $OUTPUT --imgsz $IMGSZ 

#导出onnx时同时对模型进行量化(PTQ)
python ./tools/torch2onnx.py --model $MODEL_PATH --output $OUTPUT --imgsz $IMGSZ --quantize
```

##### 参数解释:

- `$MODEL_PATH`: 训练完成后在相应文件夹下产生的模型权重文件`.pth`后缀。
- `$OUTPUT`:导出onnx的文件名，其路径在`$MODEL_PATH`下，与原模型同路径。
- `$IMGSZ`:模型所输入数据大小，图片数据则为宽高，音频数据则为长度。

## 相关教程

1. 对于导出除ONNX格式外的其他格式可参考教程[文档](./docs/zh_cn/tutorials)
2. 如何在[colab]()中使用[aroboflow](https://app.roboflow.com/)数据集训练可参见[相关教程](./docs/zh_cn/tutorials/)
3. [模型量化](./docs/zh_cn/tutorials/quantize.md)
4. [更多相关数据集结构](./docs/zh_cn/tutorials/datasets_config.md)
5. [相关工具的使用](./docs/zh_cn/tutorials/use_tools.md)

## FAQ

对于在环境配置与训练过程中可能出现的问题可先查看[相关问题解决文档](./docs/zh_cn/faq.md)
查看。若没能解决您的问题可提出[issue](https://github.com/Seeed-Studio/edgelab/issues)，
我们会尽快为你解决。

## 许可证

edgelab 目前以 Apache 2.0 的许可证发布，但是其中有一部分功能并不是使用的 Apache2.0 许可证，我们在[许可证](./LICENSES.md)
中详细地列出了这些功能以及他们对应的许可证，如果您正在从事盈利性活动，请谨慎参考此文档。




