# PyTorch 导出 ONNX

本章将介绍如何将 PyTorch 模型转换为 ONNX 模型。

## 准备工作

### 环境配置

首先，与[训练过程](../training/overview.md)类似的，我们在模型转换阶段也要求您在**虚拟环境**中完成。在 `sscma` 虚拟环境中，请确定[安装指南 - 先决条件 - 安装额外依赖项](../../introduction/installation.md#step-4-%E5%AE%89%E8%A3%85%E9%A2%9D%E5%A4%96%E7%9A%84%E4%BE%9D%E8%B5%96%E9%A1%B9-%E5%8F%AF%E9%80%89)已经完成。

::: tip

如果您配置了虚拟环境但并未激活，您可以使用以下命令激活虚拟环境:

```sh
conda activate sscma
```

:::

### 模型与权重

接下来，还需要准备好 PyTorch 模型和该模型的权重。关于模型，在[模型配置](../config.md)文件中我们已经预先配置。关于模型权重，您可以参考以下步骤来获取模型权重:

- 参考[模型训练](../training/overview.md)部分文档，选择一个模型自行训练得到模型权重。

- 或在我们的 [GitHub Releases - Model Zoo](https://github.com/Seeed-Studio/SSCMAreleases/tag/model_zoo) 中下载官方预训练的权重。

## 模型导出

关于模型转换导出，相关的工具脚本指令和一些常用参数已经列出:

```sh
python3 tools/export.py \
    "<CONFIG_FILE_PATH>" \
    "<CHECKPOINT_FILE_PATH>" \
    --target "<TARGETS>"
```

### 导出示例

以下是一些模型的转换导出示例，仅供参考:

::: code-group

```sh [FOMO Model Conversion]
python3 tools/export.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint)" \
    --target onnx \
    --cfg-options \
        data_root='datasets/mask'
```

```sh [PFLD Model Conversion]
python3 tools/export.py \
    configs/pfld/pfld_mbv2n_112.py \
    "$(cat work_dirs/pfld_mbv2n_112/last_checkpoint)" \
    --target onnx \
    --cfg-options \
        data_root='datasets/meter'
```

```sh [YOLOv5 Model Conversion]
python3 tools/export.py \
    configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
    "$(cat work_dirs/yolov5_tiny_1xb16_300e_coco/last_checkpoint)" \
    --target onnx \
    --cfg-options \
        data_root='datasets/digital_meter'
```

:::

## 模型验证

[SSCMA](https://github.com/Seeed-Studio/SSCMA) 会借助一些工具对模型进行一些优化，如模型的剪枝、蒸馏等，虽然我们在训练过程中已经对模型权重进行了测试和评估，我们建议您对导出后的模型进行再次验证。

```sh
python3 tools/inference.py \
    "<CONFIG_FILE_PATH>" \
    "<CHECKPOINT_FILE_PATH>" \
    --show \
    --cfg-options "<CFG_OPTIONS>"
```

::: tip

对于支持的更多参数，请参考代码源文件 `tools/inference.py` 或运行命令 `python3 tools/inference.py --help`。

:::

### 评估示例

::: code-group

```sh [FOMO Model Validation]
python3 tools/inference.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint | sed -e 's/.pth/.onnx/g')" \
    --show \
    --cfg-options \
        data_root='datasets/mask'
```

```sh [PFLD Model Validation]
python3 tools/inference.py \
    configs/pfld/pfld_mbv2n_112.py \
    "$(cat work_dirs/pfld_mbv2n_112/last_checkpoint | sed -e 's/.pth/.onnx/g')" \
    --show \
    --cfg-options \
        data_root='datasets/meter'
```

```sh [YOLOv5 Model Validation]
python3 tools/inference.py \
    configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
    "$(cat work_dirs/yolov5_tiny_1xb16_300e_coco/last_checkpoint | sed -e 's/.pth/.onnx/g')" \
    --show \
    --cfg-options \
        data_root='datasets/digital_meter'
```

:::
