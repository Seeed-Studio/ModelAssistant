# 快速入门

在[概述](./what_is_sscma)中，我们已经介绍了 [SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 提供的功能和特性。考虑到 [SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 被划分为多个不同的模块，每个模块完成其对应的任务，我们建议按照以下步骤快速入门。

::: tip
我们建议 [SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 的所有初学者都从[入门指南](#getting-started)开始学习，如果您对 [SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 或 [OpenMMLab](https://github.com/open-mmlab) 已经熟悉，并且希望尝试在边缘计算设备上部署、修改现有的神经网络或使用自定义数据集进行训练，您可以直接参考[高级用法](#advanced)。
:::

现在，您可以在Google Colab上尝试使用[SenseCraft Model Assistant Colab示例](https://github.com/Seeed-Studio/SSCMA/tree/main/notebooks)，无需在计算机上安装[SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA)。

## 入门指南

1. 首先，参考[安装指南](./installation.md)配置 [SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 的运行环境。

2. 然后，熟悉 [SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 的基本用法：

   - **模型训练**，请参考[模型训练](../tutorials/training/overview)以了解如何使用 [SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 进行模型训练。我们建议您从示例中选择一个模型进行训练。

   - **模型导出**。完成模型训练后，为了在边缘计算设备上部署，首先需要导出模型。有关模型导出的教程，请参考[模型导出](../tutorials/export/overview)。

   - **模型验证**。模型验证可在训练或导出后进行。前者验证神经网络和训练结果的正确性，而后者主要验证导出模型的正确性，以便后续在边缘计算设备上进行部署和调试。上述两个步骤的文档中已经提供了一些模型验证的方法。

## 高级用法

- **模型部署**。如果您想将导出的训练模型部署在边缘计算设备上，请参考[ESP32部署示例](../deploy/esp32/deploy)或[Grove Vision AI部署示例](../deploy/grove/deploy)。

- **自定义数据集**。如果您想在自定义数据集上进行训练，请参考[数据集](../tutorials/datasets)。

- **自定义模型**。如果您想修改现有的神经网络或设计自己的神经网络，请参考[模型配置](../tutorials/config)。

## 必备知识

- 📸 计算机视觉：

  计算机视觉的基础建立在数字图像处理之上。因此，您需要先学习数字图像处理的基础知识。然后可以继续学习计算机视觉主题，如模式识别和三维几何。您需要了解线性代数，以便能够充分理解计算机视觉中的某些概念，如降维。在理解计算机视觉基础知识之后，您还应该在深度学习方面建立知识，特别是卷积神经网络（CNN）方面的知识。

- 💻 编程：

  对于设计和原型制作来说，Python就足够了，但如果您想进行嵌入式开发工作，您还应该熟悉C/C++。

- 🧰 工具：

  OpenCV是计算机视觉的主要工具，Numpy是数据处理和分析的重要工具。您必须熟悉它们。您应该了解可用的工具以及如何使用它们。您还需要熟悉深度学习框架，可以从最容易学习的Keras开始，然后学习Tensorflow或PyTorch。