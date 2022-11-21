## 配置环境

运行环境需要 [PyTorch](https://pytorch.org/get-started/locally/) 和以下 OpenMMLab 仓库:

- [MIM](https://github.com/open-mmlab/mim): 一个用于管理 OpenMMLab 包和实验的命令行工具
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱和基准测试。除了分类任务，它同时用于提供多样的主干网络
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱和基准测试
- [MMDPose](https://github.com/open-mmlab/mmpose): OpenMMLab 检测工具箱和基准测试

本项目环境的配置可在ubuntu20.04上使用脚本自动完成，使用其他系统的可选择手动安装。在ubuntu上可通过以下命令即可配置完成相关环境。

```shell
python3 tools/env_config.py
```

- 对于上述命令对于国内ip会自动选择镜像，部分git仓库可能下载较慢或失败，可多次运行或者使用代理完成安装。

如果需要使用GPU进行训练，可查看GPU训练相关环境[配置教程](./docs/zh_cn/get_started.md)

<details>
<summary>Windows</summary>

将本仓库克隆至本地后进入本项目文件夹，同时将本项目文件夹路径添加至环境变量中，变量名为PYTHONPATH，添加完成后可执行以下命令查看是否添加成功。

```bash
set PYTHONPATH
```

若显示本项目地址路径表明添加成功。
</details>

<details>
<summary>Linux</summary>

同样需要将本项目的文件路径添加至系统环境变量中，变量名为PYTHONPATH，其可通过修改~/.bashrc 文件以保证在后续新终端中可用。
在终端中依次执行以下命令即可：

```bash
echo export PYTHONPATH=`pwd`:\$PYTHONPATH >> ~/.bashrc
source ~/.bashrc
```

</details>

## 开始使用

1. 首先需要确定所做的任务类型，属于目标检测、分类、或回归；确定后可根据需要选择需要的模型，并确定模型的配置文件。
2. 这里我们以端到端的音频分类模型为例演示如何训练speechcommand数据集，以及导出onnx和ncnn。

### 1.修改配置文件和激活环境

1. 在[configs](./configs)文件夹下寻找所需要修改的[配置文件](./configs/audio_classify/ali_classiyf_small_8k_8192.py)。
2. 在执行完`tools/env_config.py`的脚本后，程序已经创建了一个名为`edgelab`的虚拟环境，并安装完所有依赖。此时只需激活环境即可，如：

```shell
conda activate edgelab
```

### 2.训练

1.在配置好的环境终端中执行以下命令即可开始训练人体检测模型。

- **提示：** 虚拟换环境需要启动对应的虚拟环境

```shell
python tools/train.py mmcls $CONFIG_PATH --workdir=$WORKERDIR --gpus=1 #使用cpu可设置为0
```

#### 参数解释：

- `$CONFIG_PATH`需替换为你所使用的模型配置文件本示例中为[ali_classiyf_small_8k_8192.py](./configs/audio_classify/ali_classiyf_small_8k_8192.py)的路径。
- `$WORKERDIR`为训练过程产生的日志文件和权重文件保存的文件夹名称，默认为`worke_dir`
- `--gpus=1`表示使用一块GPU训练，若使用CPU进行训练可设置参数为`--gpus=0`

2.在模型训练完成后会在`$WORKERDIR`文件夹下产生相应的日志文件和模型权重文件(后缀为`.pth`)。

**提示：** 对于更多训练任务的使用可查看更多[训练示例](./docs/zh_cn/train_example.md)

### 3.导出ONNX

在模型训练完成后可将pth文件导出到onnx文件格式，并通过onnx转为其他想要使用的格式。
假设此时环境在本项目路径下，可通过执行如下命令导出刚才训练的人体检测模型至onnx格式。

```shell
#导出onnx时不对模型进行量化
python ./tools/torch2onnx.py  --config ./configs/audio_classify/ali_classiyf_small_8k_8192.py --checkpoint work_dirs/best.pth  --shape $IMGSZ --task mmcls --audio 
```

##### 参数解释:

- `--config`:模型训练相关配置文件的路径。
- `--checkpoint`: 训练完成后在相应文件夹下产生的模型权重文件路径(`.pth`后缀)。
- `$OUTPUT`:导出onnx的文件名，其路径在`$MODEL_PATH`下，与原模型同路径。
- `$IMGSZ`:模型所输入数据大小，图片数据则为宽高，音频数据则为长度(使用音频时，需要添加--audio参数)。

#导出ncnn
- 这里需要先将模型导出onnx后方可进行。
```shell
python ./tools/export_qiantize.py --onnx $ONNX_PATH --type $TYPE

```
##### 参数解释：
- `--$ONNX_PARH`:为模型导出的onnx格式的权重文件。
- `--$TYPE`:为需要将onnx模型导出到什么样的格式可选参数有(onnx_fp16, onnx_quan_st, onnx_quan_dy, ncnn, ncnn_fp16, ncnn_quan)。

 **提示：** `--type` 参数使用`onnx_quan_st`和`ncnn_quan` 时需要指定校验数据集的文件夹路径如:

 ```shell
 python ./tools/export_qiantize.py --onnx $ONNX_PATH --type onnx_quan_st --images ./img_e
 ```


## 相关教程

1. 对于导出除ONNX格式外的其他格式可参考教程[文档](./docs/zh_cn/tutorials)
2. 如何在[colab]()中使用[aroboflow](https://app.roboflow.com/)数据集训练可参见[相关教程](./docs/zh_cn/tutorials/)
3. [模型量化](./docs/zh_cn/tutorials/quantize.md)
4. [更多相关数据集结构](./docs/zh_cn/tutorials/datasets_config.md)
5. [相关工具的使用](./docs/zh_cn/tutorials/use_tools.md)