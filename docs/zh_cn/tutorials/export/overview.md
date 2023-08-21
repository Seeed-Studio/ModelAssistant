# 模型导出

EdgeLab 目前支持以下模型导出方式，您可以参考对应的教程，完成模型的导出，然后将导出的模型投入部署。

::: tip
默认情况下，会同时导出 ONNX 和 TFLite 模型，如果您只需要导出其中一个，可以使用 `--targets` 参数指定导出的模型类型，例如 `--targets onnx` 或 `--targets tflite`。
:::

- [PyTorch 转 ONNX](./pytorch_2_onnx.md): 将 PyTorch 模型和 `.pth` 权重转换为 ONNX 模型 `.onnx`

- [PyTorch 转 TFLite](./pytorch_2_tflite.md): 将 PyTorch 模型和 `.pth` 权重转换为 TFLite 模型 `.tflite`

::: tip

在开始导出前，您需要先完成[模型训练](../training/overview)部分内容并获得 `.pth` 模型权重。

:::

### 参数说明

您需要将以下部分参数根据实际情况进行替换，各个不同参数的具体说明如下:

```sh
python3 tools/export.py --help

# Convert and export PyTorch model to TFLite or ONNX models

# positional arguments:
#   config                the model config file path
#   checkpoint            the PyTorch checkpoint file path
#   targets               the target type of model(s) to export e.g. tflite onnx

# optional arguments:
#   -h, --help            show this help message and exit
#   --targets TARGETS [TARGETS ...]
#                         the target type of model(s) to export e.g. tflite onnx
#   --precisions PRECISIONS [PRECISIONS ...]
#                         the precisions exported model, e.g. 'int8', 'uint8', 'int16', 'float16' and 'float32'
#   --work_dir WORK_DIR, --work-dir WORK_DIR
#                         the directory to save logs and models
#   --output_stem OUTPUT_STEM, --output-stem OUTPUT_STEM
#                         the stem of output file name (with path)
#   --device DEVICE       the device used for convert & export
#   --input_shape INPUT_SHAPE [INPUT_SHAPE ...], --input-shape INPUT_SHAPE [INPUT_SHAPE ...]
#                         the shape of input data, e.g. 1 3 224 224
#   --input_type {audio,image,sensor}, --input-type {audio,image,sensor}
#                         the type of input data
#   --cfg_options CFG_OPTIONS [CFG_OPTIONS ...], --cfg-options CFG_OPTIONS [CFG_OPTIONS ...]
#                         override some settings in the used config, the key-value pair in 'xxx=yyy' format will be merged into config file
#   --simplify SIMPLIFY   the level of graph simplification, 0 means disable, max: 5
#   --opset_version OPSET_VERSION, --opset-version OPSET_VERSION
#                         ONNX: operator set version of exported model
#   --dynamic_export, --dynamic-export
#                         ONNX: export with a dynamic input shape
#   --algorithm {l2,kl}   TFLite: conversion algorithm
#   --backend {qnnpack,fbgemm}
#                         TFLite: converter backend
#   --calibration_epochs CALIBRATION_EPOCHS, --calibration-epochs CALIBRATION_EPOCHS
#                         TFLite: max epoches for quantization calibration
#   --mean MEAN [MEAN ...]
#                         TFLite: mean for model input (quantization), range: [0, 1], applied to all channels, using the average if multiple values are provided
#   --mean_and_std MEAN_AND_STD [MEAN_AND_STD ...], --mean-and-std MEAN_AND_STD [MEAN_AND_STD ...]
#                         TFLite: mean and std for model input(s), default: [((0.0,), (1.0,))], calculated on normalized input(s), applied to all channel(s), using the average if multiple values are provided
```
