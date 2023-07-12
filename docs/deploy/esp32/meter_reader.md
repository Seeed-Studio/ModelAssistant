# Meter Reader with ESP32

This tutorial will demonstrate the development process of meter reader using EdgeLab based on ESP32.

::: tip

Before starting, we recommend that you should read [ESP32 - Deploy](./deploy.md) first.

:::

## Preparation

Please refer to [ESP32 - Deploy - Prerequisites](./deploy.md#prerequisites).

## Train Model

The meter reading feature is based on the PFLD model, in this step you need a PFLD model weight with the suffix `.pth`, you have two ways to get the model weight.

- Download the pre-trained model from our [Model Zoo](https://github.com/Seeed-Studio/edgelab-model-zoo).

- Refer to [Training - PFLD Models](../../tutorials/training/pfld.md) to train the PFLD model and get the model weights using PyTorch and EdgeLab by yourself.

## Export Model

Since the trained model is not suitable for running directly on edge computing devices, we need to export it to a TFLite format with a `.tflite` suffix, and you have two ways to get the exported model (with model weights contained).

- Download the exported TFLite model from our [Model Zoo](https://github.com/Seeed-Studio/edgelab-model-zoo).

- Refer to [Export - PyTorch to TFLite](../../tutorials/export/pytorch_2_tflite.md) to convert the PFLD model from PyTorch format to TFLite format by yourself.

## Convert Model

After completing [Export Model](#export-model), we need a further process to convert it to a format that supported by embedded devices.

- Go to the `examples/esp32` directory (run at the root of the EdgeLab project):

  ```sh
  cd examples/esp32
  ```

- Convert the TFLite model to binary file:

  ```sh
  python3 tools/tflite2c.py --input <TFLITE_MODEL_PATH> --name fomo --output_dir components/modules/model --classes='("unmask", "mask")'
  ```

::: tip

You need to replace `<TFLITE_MODEL_PATH>` with the path of the TFLite model obtained in the [Export Model](#export-model) step, the final C file will be exported to the `components/modules/model` directory in the `EdgeLab/example/esp32` directory by default.

:::

## Deploy Model

This is the last and most important step to complete the meter reading, in this step you need to compile and flash the firmware to the ESP32 MCU. Please refer to [ESP32 - Deployment - Compile and Deploy](./deploy.md#compile-and-deploy) to complete the deployment of the model.

## Run Example

![PFLD Meter Reader](/static/esp32/images/pfld_meter.gif)
