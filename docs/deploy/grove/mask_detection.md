# Mask Detection with Grove - Vision AI

This tutorial will demonstrate the development process of mask detection using [ModelAssistant](https://github.com/Seeed-Studio/ModelAssistant)  based on Grove - Vision AI module.

::: tip

Before starting, we recommend that you should read [Grove - Deploy](./deploy.md) first.

:::

## Preparation

Please refer to [Grove - Deploy - Prerequisites](./deploy.md#prerequisites).

## Train Model

The mask detection feature is based on the FOMO model, in this step you need a FOMO model weight with the suffix `.pth`, you have two ways to get the model weight.

- Download the pre-trained model from our [Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo).

- Refer to [Training - FOMO Models](../../tutorials/training/fomo.md) to train the FOMO model and get the model weights using PyTorch and [ModelAssistant](https://github.com/Seeed-Studio/ModelAssistant)  by yourself.

## Export Model

Since the trained model is not suitable for running directly on edge computing devices, we need to export it to a TFLite format with a `.tflite` suffix, and you have two ways to get the exported model (with model weights contained).

- Download the exported TFLite model from our [Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo).

- Refer to [Export - PyTorch to TFLite](../../tutorials/export/pytorch_2_tflite.md) to convert the FOMO model from PyTorch format to TFLite format by yourself.

## Deploy Model

This is the last and most important step to complete the mask detection, in this step you need to compile and flash the firmware to the Grove - Vision AI module. Please refer to [Grove - Deployment - Compile and Deploy](./deploy.md#compile-and-deploy) to complete the deployment of the model.
