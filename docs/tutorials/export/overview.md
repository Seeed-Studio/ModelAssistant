# Model Export

EdgeLab currently supports the following methods to convert and export models. You can refer to the corresponding tutorials to complete the model export, and then put the exported model into deployment.

- [PyTorch to ONNX](./pytorch_2_onnx.md): Converts PyTorch model and `.pth` weights to ONNX model `.onnx`

- [PyTorch to TFLite](./pytorch_2_tflite.md): Converts PyTorch model and `.pth` weights to TFLite model `.tflite`

::: tip

Before you can start exporting models, you need to complete the [Training](../training/overview) section and obtain model weights `.pth` file before start exporting.

:::

## Parameter Descriptions

For more parameters for model exporting, you can refer the code below.

```sh
python3 tools/export.py --help
# positional arguments:
#   config                the model config file path
#   checkpoint            the PyTorch checkpoint file path
#   targets               the target type of model(s) to export e.g. tflite onnx
# optional arguments:
#   -h, --help            show this help message and exit
#   --precisions PRECISIONS [PRECISIONS ...]
#                         the precisions exported model, e.g. 'int8', 'uint8', 'int16', 'float16' and 'float32'
#   --work_dir WORK_DIR   the directory to save logs and models
#   --output_stem OUTPUT_STEM
#                         the stem of output file name (with path)
#   --device DEVICE       the device used for convert & export
#   --input_shape INPUT_SHAPE [INPUT_SHAPE ...]
#                         the shape of input data, e.g. 1 3 224 224
#   --input_type {audio,image,sensor}
#                         the type of input data
#   --cfg_options CFG_OPTIONS [CFG_OPTIONS ...]
#                         override some settings in the used config, the key-value pair in 'xxx=yyy' format will be merged into config file
#   --simplify SIMPLIFY   the level of graph simplification, 0 means disable, max: 5
#   --opset_version OPSET_VERSION
#                         ONNX: operator set version of exported model
#   --dynamic_export      ONNX: export with a dynamic input shape
#   --algorithm {l2,kl}   TFLite: conversion algorithm
#   --backend {qnnpack,fbgemm}
#                         TFLite: conveter backend
#   --epoch EPOCH         TFLite: max epoches for quantization calibration
#   --mean MEAN [MEAN ...]
#                         TFLite: mean for model input (quantization), range: [0, 1], applied to all channels, using the average if multiple values are provided
#   --mean_and_std MEAN_AND_STD [MEAN_AND_STD ...]
#                         TFLite: mean and std for model input(s), defalut: [((0.0,), (1.0,))], calculated on normalized input(s), applied to all channel(s), using the average if multiple values are provided
```
