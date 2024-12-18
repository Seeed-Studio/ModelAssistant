<div align="center">
  <img width="20%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-Hero.png"/>
</div>

# 项目简介

Seeed SenseCraft Model Assistant (SSCMA)  是一个专注于嵌入式人工智能的开源项目。我们针对实际场景优化，致力于提供良好的用户的体验，在嵌入式设备上实现更快速、更准精确的推理。

## 包含哪些内容？

目前，我们支持以下几个方向的算法：

### 🔍 异常检测

在现实世界中，异常数据通常很难识别，即使能够识别，也需要很高的成本。异常检测算法以低成本的方式收集正常数据，将正常数据之外的任何内容视为异常。

### 👁️ 计算机视觉

我们提供了许多计算机视觉算法，如目标检测、图像分类、图像分割和姿态估计。然而，这些算法无法在低成本硬件上运行。[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 优化了这些计算机视觉算法，在低端设备上实现了良好的运行速度和准确性。

### ⏱️ 特定场景

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 为特定的生产环境提供定制化场景，例如模拟仪器的识别、传统数字仪表和音频分类。我们将继续在未来为指定场景添加更多算法。


## 特点

### 🤝 用户友好

我们提供了一个用户友好的平台，让用户可以轻松地对收集的数据进行训练，并通过在训练过程中生成的可视化结果更好地了解算法的性能。

### 🔋 低计算能力高性能模型

我们专注于边缘端 AI 算法研究，算法模型可部署在微处理器上，类似于 [ESP32](https://www.espressif.com/en/products/socs/esp32)、一些 [Arduino](https://arduino.cc) 开发板，甚至嵌入式 SBC（如 [Raspberry Pi](https://www.raspberrypi.org) ）等设备上。

### 🗂️ 支持多种模型导出格式

我们致力于解决端侧部署模型的碎片化问题，支持多种模型导出格式。例如，[TensorFlow Lite](https://www.tensorflow.org/lite) 主要用于微控制器，而 [ONNX](https://onnx.ai) 在嵌入式 Linux 设备更加常见。还有一些和硬件强相关的，特殊格式，如 [TensorRT](https://developer.nvidia.com/tensorrt)、[OpenVINO](https://docs.openvino.ai)、[Vela](https://developer.arm.com/documentation/109267/latest/Tool-support-for-the-Arm-Ethos-U-NPU/Ethos-U-Vela-compiler)、[HailoRT](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/#sw-hailort) 等，这些格式已经得到了良好支持。

### 🚀 面向未来

在人工智能向边缘端发展的过程中，低延迟、低功耗、低成本相对与高精度、考效率是之间的平衡一直是一个难题，但这是未来必然的趋势。因此诞生了许多特殊硬件加速器（NPU、TPU、VPU LPU 等），但这些硬件往往对模型的支持有限，或者对一个模型的结构、算子有特殊的要求。在实际应用场景中，一个相同的模型可能需要在不同的硬件上部署，这不仅仅需要我们对模型进行转换，还需要修改部分结构，这需要从部署端出发、将算法以良好支持的形式定义、在海量的数据上训练、调优，形成闭环，才能真正实现边缘端 AI 的落地。我们将持续关注这个领域，通过 QAT、MLIR、TVM 等技术，为用户提供更好的支持。


## SSCMA 工具链

SSCMA 提供了完整的工具链，让用户可以轻松地在低成本硬件上部署 AI 模型，包括：

- [SSCMA-Model-Zoo](https://github.com/Seeed-Studio/sscma-model-zoo) SSCMA 模型库为您提供了一系列针对不同应用场景的预训练模型。
- [SSCMA-Micro](https://github.com/Seeed-Studio/SSCMA-Micro) 一个跨平台的框架，用于在微控制器设备上部署和应用 SSCMA 模型。
- [Seeed-Arduino-SSCMA](https://github.com/Seeed-Studio/Seeed_Arduino_SSCMA) 支持 SSCMA-Micro 固件的 Arduino 库。
- [SSCMA-Web-Toolkit](https://seeed-studio.github.io/SenseCraft-Web-Toolkit) 一个基于 Web 的工具，用于更新设备固件、SSCMA 模型和参数。
- [Python-SSCMA](https://github.com/Seeed-Studio/python-sscma) 用于与微控制器进行交互的 Python 库，使用 SSCMA-Micro，并用于更高级别的深度学习应用。


## 应用示例

SSCMA 可以应用于许多领域，解決各种实际问题，賦能并提高生产效率，以下是一些示例：

### 目标检测

目标检测是一种计算机视觉技术，旨在识别和定位图像或视频中特定对象。它不仅可以识别出图像中存在的物体类别（如人、车、动物等），还可以确定这些物体在图像中的位置，通常以边界框的形式表示。

<div align="center"><img width="800" src="https://files.seeedstudio.com/sscma/docs/static/esp32/images/person_detection.png"/></div>

### 指针表计

使用人工智能技术识别和读取指针表计（如电表、水表等）上的数值。

<div align="center"><img width="800" src="https://files.seeedstudio.com/sscma/docs/static/grove/images/pfld_meter.gif"/></div>

### 数字表计

类似于指针表计，使用人工智能技术识别和读取数字表计（如电子表、数字电表等）上的数值。

<div align="center"><img width="800" src="https://files.seeedstudio.com/sscma/docs/static/grove/images/digital_meter.gif"/></div>

更多的应用请参考 [Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo)。
