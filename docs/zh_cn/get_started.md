## 配置环境

运行环境需要 [PyTorch](https://pytorch.org/get-started/locally/) 和以下 OpenMMLab 第三方库:

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱和基准测试。除了分类任务，它同时用于提供多样的主干网络
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱和基准测试
- [MMDPose](https://github.com/open-mmlab/mmpose): OpenMMLab 检测工具箱和基准测试

本项目环境的配置可在ubuntu20.04上使用脚本自动完成，使用其他系统的可选择手动安装。在ubuntu上可通过以下命令即可配置所有相关环境。

```shell
python3 tools/env_config.py
```

**提示：** 以上环境配置时长根据设网络环境会有差异。

在上诉步骤执行完毕后，在文件~/.bashrc中已经添加了各种所需的环境变量。同时建立了一个名为edgelab的conda虚拟环境，相关依赖也被安装在虚拟环境中，但此时并未激活。可通过一下命令激活conda、虚拟环境和其他相关环境变量。

```bash
source ~/.bashrc
conda activate edgelab
```

## 简单使用

- 首先需要确定所做的任务类型，属于目标检测、分类、或关键点检测；确定后可根据需要选择需要的模型，并确定模型的配置文件。
- 这里我们以端到端的音频分类模型为例演示如何训练speechcommand数据集，以及如何导出onnx和ncnn。

### 1.确定配置文件

- 在[configs](../../configs)文件夹下寻找所需要的[配置文件](../../configs//audio_classify/ali_classiyf_small_8k_8192.py)。

**提示：** 所有模型的配置文件都在[configs](../../configs)文件夹下。

### 2.训练

- 在已激活的虚拟环境终端中执行以下命令即可开始训练端到端的语音分类模型。

```shell
python tools/train.py mmcls $CONFIG_PATH --workdir=$WORKERDIR --gpus=1 #使用cpu可设置为0
```

以上过程中在没有修改配置文件数据集路径的情况下，程序会自动下载speechcommand数据集。

#### 参数解释

- `$CONFIG_PATH`需替换为你所使用的模型配置文件本示例中为[ali_classiyf_small_8k_8192.py](../../configs/audio_classify/ali_classiyf_small_8k_8192.py)的路径。
- `$WORKERDIR`为训练过程产生的日志文件和权重文件保存的文件夹名称，默认为`worke_dir`
- `--gpus=1`表示使用一块GPU训练，若使用CPU进行训练可设置参数为`--gpus=0`

2.在模型训练完成后会在`$WORKERDIR`文件夹下产生相应的日志文件和模型权重文件(后缀为`.pth`)。

**提示：** 对于更多训练任务的使用可查看更多[训练示例](../zh_cn/train_example.md)

### 3.导出ONNX

在模型训练完成后可将pth文件导出到onnx文件格式，并通过onnx转为其他想要使用的格式。
假设此时环境在本项目路径下，可通过执行如下命令导出刚才训练的人体检测模型至onnx格式。

```shell
python ./tools/torch2onnx.py  --config ./configs/audio_classify/ali_classiyf_small_8k_8192.py --checkpoint work_dirs/best.pth  --shape $IMGSZ --task mmcls --audio 
```

##### 参数解释

- `--config`:模型训练相关配置文件的路径。
- `--checkpoint`: 训练完成后在相应文件夹下产生的模型权重文件路径(`.pth`后缀)。
- `$OUTPUT`:导出onnx的文件名，其路径在`$MODEL_PATH`下，与原模型同路径。
- `$IMGSZ`:模型所输入数据大小，图片数据则为宽高，音频数据则为长度(使用音频时，需要添加`--audio`参数)。

### 4.导出ncnn

- 这里需要先将模型导出onnx后方可进行。

```shell
python ./tools/export_qiantize.py --onnx $ONNX_PATH --type $TYPE

```

更多有关onnx与ncnn量化导出可查看[详细教程](./tutorials/onnx2xxx.md)

##### 参数解释

- `--$ONNX_PARH`:为模型导出的onnx格式的权重文件(上面的`$OUTPUT`参数)。
- `--$TYPE`:为需要将onnx模型导出到什么样的格式可选参数有(onnx_fp16, onnx_quan_st, onnx_quan_dy, ncnn, ncnn_fp16, ncnn_quan)。

 **提示：** `--type` 参数使用`onnx_quan_st`和`ncnn_quan` 时需要指定校验数据集的文件夹路径如:

 ```shell
 python ./tools/export_qiantize.py --onnx $ONNX_PATH --type onnx_quan_st --images ./img_e
 ```

## 相关教程

1. 对于导出除ONNX格式外的其他格式可参考教程[文档](./docs/zh_cn/tutorials)
2. 如何在[colab]()中使用[aroboflow](https://app.roboflow.com/)数据集训练可参见[相关教程](./tutorials/)
3. [模型导出与量化](./tutorials/onnx2xxx.md)
4. [更多相关数据集结构](./tutorials/datasets_config.md)
5. [相关工具的使用](./tutorials/use_tools.md)
