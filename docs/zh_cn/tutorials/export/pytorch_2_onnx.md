# PyTorch 导出 ONNX (实验性支持)

本章将介绍如何将 PyTorch 模型转换为 ONNX 模型。


## 准备工作

### 环境配置

首先，与[训练过程](../training/overview.md)类似的，我们在模型转换阶段也要求您在**虚拟环境**中完成。在 `edgelab` 虚拟环境中，请确定[安装指南 - 先决条件 - 安装额外依赖项](../../introduction/installation.md#step-4-安装额外的依赖项-可选)已经完成。

::: tip

如果您配置了虚拟环境但并未激活，您可以使用以下命令激活虚拟环境:

```sh
conda activate edgelab
```

:::

### 模型与权重

接下来，还需要准备好 PyTorch 模型和该模型的权重。关于模型，在[模型配置](../config.md)文件中我们已经预先配置。关于模型权重，您可以参考以下步骤来获取模型权重:

- 参考[模型训练](../training/overview.md)部分文档，选择一个模型自行训练得到模型权重。

- 或在我们的 [GitHub Releases - Model Zoo](https://github.com/Seeed-Studio/EdgeLab/releases/tag/model_zoo) 中下载官方预训练的权重。


## 模型导出

关于模型转换导出，相关的工具脚本指令和一些常用参数已经列出:

```sh
python3 tools/torch2onnx.py \
    <TASK> \
    <CONFIG_FILE_PATH> \
    --checkpoint <CHECKPOINT_FILE_PATH> \
    --simplify <SIMPLIFY> \
    --shape <SHAPE>
```

### 导出参数

您需要将以上参数根据实际情况进行替换，各个不同参数的具体说明如下:

- `<TASK>` - 模型的类型，可选参数: `['det', 'cls', 'pose']`

- `<CONFIG_FILE_PATH>` - 模型配置文件的路径

- `<CHECKPOINT_FILE_PATH>` - 模型权重文件的路径

- `<SIMPLIFY>` - 是否简化模型，默认 `False`

- `<SHAPE>` - 模型的输入张量的维度，默认 `112`

::: tip

对于支持的更多参数，请参考代码源文件 `tools/torch2onnx.py`。

:::

### 导出示例

以下是一些模型的转换导出示例，仅供参考:

::: code-group

```sh [FOMO 模型导出]
python3 tools/torch2onnx.py \
    det \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    --checkpoint "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint)" \
    --shape 96
```

```sh [PFLD 模型导出]
python3 tools/torch2onnx.py \
    pose \
    configs/pfld/pfld_mv2n_112.py \
    --checkpoint "$(cat work_dirs/pfld_mv2n_112/last_checkpoint)" \
    --shape 112
```

:::


## 模型验证

由于在导出模型的过程中，EdgeLab 会借助一些工具对模型进行一些优化，如模型的剪枝、蒸馏等，虽然我们在训练过程中已经对模型权重进行了测试和评估，我们建议您对导出后的模型进行再次验证。

```sh
python3 tools/test.py \
    <TASK> \
    <CONFIG_FILE_PATH> \
    <CHECKPOINT_FILE_PATH> \
    --out <OUT_FILE_PATH> \
    --work-dir <WORK_DIR_PATH> \
    --cfg-options <CFG_OPTIONS>
```

### 参数描述

您需要将以上参数根据实际情况进行替换，各个不同参数的具体说明如下:

- `<TASK>` - 模型的类型，可选参数: `['det', 'cls', 'pose']`

- `<CONFIG_FILE_PATH>` - 模型配置文件的路径

- `<CHECKPOINT_FILE_PATH>` - 模型权重文件的路径

- `<OUT_FILE_PATH>` - (可选) 验证结果输出的文件路径

- `<WORK_DIR_PATH>` - (可选) 工作目录的路径

- `<CFG_OPTIONS>` - (可选) 配置文件参数覆写，具体请参考[模型配置 - EdgeLab 参数化配置](../config.md#edgelab-参数化配置)

::: tip

对于支持的更多参数，请参考代码源文件 `tools/test.py`。

:::

### 评估示例

::: code-group

```sh [FOMO 模型评估]
python3 tools/test.py \
    det \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint | sed -e 's/.pth/.onnx/g')" \
    --cfg-options \
        data_root='datasets/mask'
```

```sh [PFLD 模型评估]
python3 tools/test.py \
    pose \
    configs/pfld/pfld_mv2n_112.py \
    "$(cat work_dirs/pfld_mv2n_112/last_checkpoint | sed -e 's/.pth/.onnx/g')" \
    --cfg-options \
        data_root='datasets/meter'
```

:::
