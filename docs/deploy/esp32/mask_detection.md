# Mask Detection with ESP32

This tutorial will demonstrate the development process of mask detection using [ModelAssistant](https://github.com/Seeed-Studio/ModelAssistant) based on ESP32.

::: tip

Before starting, we recommend that you should read [ESP32 - Deploy](./deploy.md) first.

:::

## Preparation

Please refer to [ESP32 - Deploy - Prerequisites](./deploy.md#prerequisites).

## Train Model

The mask detection feature is based on the FOMO model, in this step you need a FOMO model weight with the suffix `.pth`, you have two ways to get the model weight.

- Download the pre-trained model from our [Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo).

- Refer to [Training - FOMO Models](../../tutorials/training/fomo.md) to train the FOMO model and get the model weights using PyTorch and [ModelAssistant](https://github.com/Seeed-Studio/ModelAssistant) by yourself.

## Export Model

Since the trained model is not suitable for running directly on edge computing devices, we need to export it to a TFLite format with a `.tflite` suffix, and you have two ways to get the exported model (with model weights contained).

- Download the exported TFLite model from our [Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo).

- Refer to [Export - PyTorch to TFLite](../../tutorials/export/pytorch_2_tflite.md) to convert the FOMO model from PyTorch format to TFLite format by yourself.

## Convert Model

After completing [Export Model](#export-model), we need a further process to convert it to a format that supported by embedded devices.

- Go to the `examples/esp32` directory (run at the root of the [ModelAssistant](https://github.com/Seeed-Studio/ModelAssistant) project):

  ```sh
  cd examples/esp32
  ```

- Convert the TFLite model to binary file:

  ```sh
  python3 tools/tflite2c.py --input <TFLITE_MODEL_PATH> --name fomo --output_dir components/modules/model --classes='("unmask", "mask")'
  ```

::: tip

You need to replace `<TFLITE_MODEL_PATH>` with the path of the TFLite model obtained in the [Export Model](#export-model) step, the final C file will be exported to the `components/modules/model` directory in the `ModelAssistant/example/esp32` directory by default.

:::

## Deploy Model

This is the last and most important step to complete the mask detection, in this step you need to compile and flash the firmware to the ESP32 MCU. Please refer to [ESP32 - Deployment - Compile and Deploy](./deploy.md#compile-and-deploy) to complete the deployment of the model.

## Run Example

![FOMO Mask Detection](/static/esp32/images/fomo_mask.gif)
