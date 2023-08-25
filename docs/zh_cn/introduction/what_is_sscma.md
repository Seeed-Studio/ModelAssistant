# 项目简介

![SSCMA-logo](/images/SSCMA-Logo.png)

[SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 是一个专注于嵌入式人工智能的开源项目。我们从 [OpenMMLab](https://github.com/open-mmlab) 优化了出色的算法，针对实际场景进行了改进，并使实现更加用户友好，从而在嵌入式设备上实现更快速、更准确的推断。

## 包含哪些内容？

目前，我们支持以下几个方向的算法：

### 🔍 异常检测
在现实世界中，异常数据通常很难识别，即使能够识别，也需要很高的成本。异常检测算法以低成本的方式收集正常数据，将正常数据之外的任何内容视为异常。

### 👁️ 计算机视觉
我们提供了许多计算机视觉算法，如目标检测、图像分类、图像分割和姿态估计。然而，这些算法无法在低成本硬件上运行。[SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 优化了这些计算机视觉算法，在低端设备上实现了良好的运行速度和准确性。

### ⏱️ 特定场景
[SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 为特定的生产环境提供定制化场景，例如模拟仪器的识别、传统数字仪表和音频分类。我们将继续在未来为指定场景添加更多算法。

## 特点

### 用户友好
[SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 提供了一个用户友好的平台，让用户可以轻松地对收集的数据进行训练，并通过在训练过程中生成的可视化结果更好地了解算法的性能。

### 低计算能力高性能模型
[SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA) 专注于边缘端AI算法研究，算法模型可部署在微处理器上，类似于[ESP32]("https://www.espressif.com/en/products/socs/esp32")、一些[Arduino]("https://arduino.cc")开发板，甚至嵌入式SBC（如 [Raspberry Pi](https://www.raspberrypi.org) ）等设备上。

### 支持多种模型导出格式
[TensorFlow Lite](https://www.tensorflow.org/lite) 主要用于微控制器，而 [ONNX](https://onnx.ai) 主要用于嵌入式Linux设备。还有一些特殊格式，如 [TensorRT](https://developer.nvidia.com/tensorrt)、[OpenVINO](https://docs.openvino.ai)，这些格式已经得到OpenMMLab的良好支持。[SenseCraft Model Assistant](https://github.com/Seeed-Studio/SSCMA)添加了用于微控制器的TFLite模型导出功能，可以直接转换为 [TensorRT]("https://developer.nvidia.com/tensorrt) 格式。