
# PFLD模型训练
本节会介绍如何在表计数据集上训练表计模型。
- [PFLD模型训练](#pfld模型训练)
    - [数据集准备](#数据集准备)
    - [配置文件](#配置文件)
    - [训练模型](#训练模型)
        - [参数描述](#参数描述)
    - [测试和评估](#测试和评估)
        - [测试](#测试)
        - [评估](#评估)
    - [提醒](#提醒)
    - [FAQs](#faqs)

## 数据集准备
我们已经准备好一个可用数据集。
- **step 1.** 点击[这里](https://1drv.ms/u/s!AqG2uRmVUhlShtIhyd_7APHXEhpeXg?e=WwGx5m)下载数据集。
- **step 2.** 解压下载数据集并记住文件路径，当修改配置文件时会使用文件路径。

## 配置文件
我们将根据需要执行的任务类型来选择配置文件，我们已经在[配置](../../../../configs/pfld/)中准备好预配置文件。

对于表计模型示例，我们使用[pfld_mv2n_112.py](../../../../configs/pfld/pfld_mv2n_112.py)配置文件，这个文件主要用于配置训练所需要的数据集，包括数据集路径。

## 训练模型
在激活的conda虚拟环境下，执行下面的命令来训练一个端到端的表计模型。

```sh
python tools/train.py \
    ${TYPE} \
    ${CONFIG_FILE} \
    --work-dir ${WORK-DIR} \
    --gpu-id ${GPU-ID} \
    --cfg-options ${CFG-OPTIONS} \
```
### 参数描述
- `${TYPE}` 训练模型类型，[`mmdet`, `mmcls`, `mmpose`]，表计模型使用`mmpose`。
- `${CONFIG_FILE}` 模型配置文件(在configs路径下)。
- `--work-dir` 用于保存当前实验的模型检查点和日志的目录。
- `gpu-id` 使用的gpu id（仅适用于非分布式训练）。
- `--cfg-options` 覆盖使用的配置文件中的一些设置，xxx=yyy格式的键值对将被合并到配置文件中。

**注意:** `--cfg-options`会需要一些参数，更多细节你可以查看[website](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html)，然后使用下面命令改变相应参数:

```sh
--cfg-options \
    data_root=${DATA-ROOT} \
    load_from=${LOAD-FROM} \
```
- `${DATA-ROOT}` 训练数据集路径。
- `${LOAD-FROM}` 从给定路径加载模型作为预训练模型，这不会恢复训练。

训练完成后，模型权重文件会保存在 **~/Edgelab/work_dir/pfld_mv2n_112/exp1/latest.pth**。请记住权重文件路径，在模型转换其他格式时会需要。

## 测试和评估

### 测试
使用下面命令测试模型：
```sh
python tools/test.py \
    mmpose \
    configs/pfld/pfld_mv2n_112.py \
    pfld_mv2n_112.pth \
    --no_show \
```

### 评估
为了导出其他模型格式来进行验证，请阅读[pytorch2onnx](../export/pytorch2onnx.md) and [pytorch2tflite](../export/pytorch2tflite.md)。

## 提醒
- 

## FAQs
- 
