# PyTorch to TFLite

This chapter will describe how to convert and export PyTorch models to TFLite models.

## Preparation

### Environment Configuration

As the [Training](../training/overview.md) step, we recommend you to do it in a **virtual environment** during the model exporting phase. In the `sscma` virtual environment, make sure that the [Installation - Prerequisites - Install Extra Dependencies](../../introduction/installation#step-4-install-extra-dependencies-optional) step has been completed.

::: tip

If you have configured a virtual environment but not activated it, you can activate it with the following command.

```sh
conda activate sscma
```

:::

### Models and Weights

You also need to prepare the PyTorch model and its weights before exporting the model. For the model, you can find it in the [Config](../config.md) section, we have already preconfigured. For the weights, you can refer to the following steps to get the model weights.

- Refer to [Training](../training/overview.md) section and choose a model, and train to get the model weights.

- Or download the [SSCMA](https://github.com/Seeed-Studio/SSCMA) official pre-trained weights from our [GitHub Releases - Model Zoo](https://github.com/Seeed-Studio/SSCMAreleases/tag/model_zoo).

::: tip

Export TFLite model requires a training set as a representative dataset, if it not found, the program will download it automatically. However, for some large datasets, this can take a long time, so please be patient.

:::

## Export Model

For model transformation (convert and export), the relevant commands with some common parameters are listed.

```sh
python3 tools/export.py \
    "<CONFIG_FILE_PATH>" \
    "<CHECKPOINT_FILE_PATH>" \
    --target tflite
```

### TFLite Export Examples

Here are some model conversion examples (`int8` precision) for reference.

::: code-group

```sh [FOMO Model Conversion]
python3 tools/export.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint)" \
    --target tflite \
    --cfg-options \
        data_root='datasets/mask'

```

```sh [PFLD Model Conversion]
python3 tools/export.py \
    configs/pfld/pfld_mbv2n_112.py \
    "$(cat work_dirs/pfld_mbv2n_112/last_checkpoint)" \
    --target tflite \
    --cfg-options \
        data_root='datasets/meter'
```

```sh [YOLOv5 Model Conversion]
python3 tools/export.py \
    configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
    "$(cat work_dirs/yolov5_tiny_1xb16_300e_coco/last_checkpoint)" \
    --target tflite
    --cfg-options \
        data_root='datasets/digital_meter'
```

:::

## Model Validation

Since in the process of exporting the model, [SSCMA](https://github.com/Seeed-Studio/SSCMA) will do some optimization for the model using some tools, such as model pruning, distillation, etc. Although we have tested and evaluated the model weights during the training process, we recommend you to validate the exported model again.

```sh
python3 tools/inference.py \
    "<CONFIG_FILE_PATH>" \
    "<CHECKPOINT_FILE_PATH>" \
    --show \
    --cfg-options "<CFG_OPTIONS>"
```

::: tip

For more parameters supported, please refer to the source code `tools/inference.py` or run `python3 tools/inference.py --help`.

:::

### Model Validation Example

Here are some examples for validating converted model (`int8` precision), for reference only.

::: code-group

```sh [FOMO Model Validation]
python3 tools/inference.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint | sed -e 's/.pth/_int8.tflite/g')" \
    --show \
    --cfg-options \
        data_root='datasets/mask'
```

```sh [PFLD Model Validation]
python3 tools/inference.py \
    configs/pfld/pfld_mbv2n_112.py \
    "$(cat work_dirs/pfld_mbv2n_112/last_checkpoint | sed -e 's/.pth/_int8.tflite/g')" \
    --show \
    --cfg-options \
        data_root='datasets/meter'
```

```sh [YOLOv5 Model Validation]
python3 tools/inference.py \
    configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
    "$(cat work_dirs/yolov5_tiny_1xb16_300e_coco/last_checkpoint | sed -e 's/.pth/_int8.tflite/g')" \
    --show \
    --cfg-options \
        data_root='datasets/digital_meter'
```

:::
