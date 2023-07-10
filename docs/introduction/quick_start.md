# Quick Start

In [Overview](./what_is_edgelab), we have introduced the functions and features provided by EdgeLab. Considering that EdgeLab is divided into multiple different modules, each module completing its corresponding tasks, we suggest following the steps below to quickly get started.

::: tip
We suggest that all beginners of EdgeLab start learning from [Getting Started](#getting-started), if you are familiar with EdgeLab or [OpenMMLab](https://github.com/open-mmlab), and you want to try to deploy on edge computing devices, modify existing neural networks, or train on user-defined data sets, you can directly refer to [Advanced](#advanced).
:::

## Getting Started

1. First, refer to the [Installation Guide](./installation.md) to configure the running environment of EdgeLab.

2. Then, familiar with the basic usage methods of EdgeLab:

   - **Model Training**, please refer to [Model Training](../tutorials/training/overview) to learn how to use EdgeLab to train a model. We suggest that you select a model from an example for training.

   - **Model Export**. After completing model training, in order to deploy on the edge-computing device, it is necessary to first export the model. For the export tutorial of the model, please refer to [Model Export](../tutorials/export/overview).

   - **Model Verification**. Model verification can be performed after training or export. The former verifies the correctness of the neural network and training results, while the latter mainly verifies the correctness of the exported model, facilitating deployment and debugging on edge computing devices later. Some methods for model validation have been provided in the documents in the above two steps.

## Advanced

- **Model Deployment**. If you want to deploy the exported training model on edge computing devices, please refer to [ESP32 Deployment Example](../examples/esp32/deploy) or [Grove Vision AI Deployment Example](../examples/grove/deploy).

- **Custom Datasets**. If you want to train on a custom dataset, please refer to [Datasets](../tutorials/datasets).

- **Custom Model**. If you want to modify an existing neural network or design your own neural network, please refer to [Model Configuration](../tutorials/config).
