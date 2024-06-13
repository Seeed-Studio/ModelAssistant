<div align="center">
  <img width="20%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-Hero.png"/>

  <h1>
      SenseCraft Model Assistant by Seeed Studio
  </h1>

[![docs-build](https://github.com/Seeed-Studio/ModelAssistant/actions/workflows/docs-build.yml/badge.svg)](https://github.com/Seeed-Studio/ModelAssistant/actions/workflows/docs-build.yml)
[![functional-test](https://github.com/Seeed-Studio/ModelAssistant/actions/workflows/functional-test.yml/badge.svg?branch=main)](https://github.com/Seeed-Studio/ModelAssistant/actions/workflows/functional-test.yml)
![GitHub Release](https://img.shields.io/github/v/release/Seeed-Studio/ModelAssistant)
[![license](https://img.shields.io/github/license/Seeed-Studio/ModelAssistant.svg)](https://github.com/Seeed-Studio/ModelAssistant/blob/main/LICENSE)
[![Average time to resolve an issue](http://isitmaintained.com/badge/resolution/Seeed-Studio/ModelAssistant.svg)](http://isitmaintained.com/project/Seeed-Studio/ModelAssistant "Average time to resolve an issue")
[![Percentage of issues still open](http://isitmaintained.com/badge/open/Seeed-Studio/ModelAssistant.svg)](http://isitmaintained.com/project/Seeed-Studio/ModelAssistant "Percentage of issues still open")

  <h3>
    <a href="https://sensecraftma.seeed.cc"> 文档 </a> |
    <a href="https://sensecraftma.seeed.cc/introduction/installation"> 安装 </a> |
    <a href="https://github.com/Seeed-Studio/ModelAssistant/tree/main/notebooks"> Colab </a> |
    <a href="https://github.com/Seeed-Studio/sscma-model-zoo"> 模型仓库 </a> |
    <a href="https://seeed-studio.github.io/SenseCraft-Web-Toolkit"> 部署 </a> -
    <a href="README.md"> English </a>
  </h3>

</div>

## 简介

**S**eeed **S**ense**C**raft **M**odel **A**ssistant 是一个专注于为嵌入式设备提供最先进的人工智能算法的开源项目。它旨在帮助开发人员和制造商轻松部署各种人工智能模型到低成本硬件上，如微控制器和单板计算机（SBCs）。

<div align="center">

<img width="98%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-Deploy.gif"/>

</div>

**在功耗低于 0.3 瓦的微控制器上的真实部署示例。*

### 🤝 用户友好

SenseCraft 模型助手提供了一个用户友好的平台，方便用户使用收集的数据进行训练，并通过训练过程中生成的可视化结果更好地了解算法的性能。

### 🔋 低计算功耗、高性能的模型

SenseCraft 模型助手专注于边缘端人工智能算法研究，算法模型可以部署在微处理器上，类似于 [ESP32](https://www.espressif.com.cn/en/products/socs/esp32)、一些 [Arduino](https://arduino.cc) 开发板，甚至在嵌入式 SBCs（如 [Raspberry Pi](https://www.raspberrypi.org) ）上。

### 🗂️ 支持多种模型导出格式

[TensorFlow Lite](https://www.tensorflow.org/lite) 主要用于微控制器，而 [ONNX](https://onnx.ai) 主要用于嵌入式Linux设备。还有一些特殊格式，如 [TensorRT](https://developer.nvidia.com/tensorrt)、[OpenVINO](https://docs.openvino.ai)，这些格式已经得到 OpenMMLab 的良好支持。SenseCraft 模型助手添加了 TFLite 模型导出功能，可直接转换为 [TensorRT](https://developer.nvidia.com/tensorrt) 和 [UF2](https://github.com/microsoft/uf2) 格式，并可拖放到设备上进行部署。

## 功能

我们已经从 [OpenMMLab](https://github.com/open-mmlab) 优化了出色的算法，针对实际场景进行了改进，并使实现更加用户友好，实现了更快、更准确的推理。目前我们支持以下算法方向:

### 🔍 异常检测

在现实世界中，异常数据通常难以识别，即使能够识别出来，也需要很高的成本。异常检测算法以低成本的方式收集正常数据，认为任何超出正常数据范围的数据都是异常的。

### 👁️ 计算机视觉

我们提供了许多计算机视觉算法，例如目标检测、图像分类、图像分割和姿态估计。但是，这些算法无法在低成本硬件上运行。SenseCraft 模型助手优化了这些计算机视觉算法，实现了较好的运行速度和准确性。

### ⏱️ 场景特定

SenseCraft 模型助手为特定的生产环境提供了定制化场景，例如模拟仪器、传统数字仪表和音频分类的识别。我们将继续在未来添加更多的指定场景算法。

## 新特性

SSCMA 一直致力于为用户提供最先进的人工智能算法，以获得最佳性能和准确性。我们根据社区反馈不断更新和优化算法，以满足用户的实际需求。以下是一些最新的更新内容:

### 🔥 YOLO-World、MobileNetV4 和更轻量的 SSCMA（即将推出）

我们正在为嵌入式设备开发最新的 [YOLO-World](https://github.com/AILab-CVC/YOLO-World)和 [MobileNetV4](https://arxiv.org/abs/2404.10518) 算法。同时，我们也正在重新设计 SSCMA，减少其依赖项，使其更加轻量级和易于使用。请密切关注最新的更新。

### YOLOv8、YOLOv8 Pose、Nvidia Tao Models 和 ByteTrack

通过 [SSCMA-Micro](https://github.com/Seeed-Studio/SSCMA-Micro)，现在您可以在微控制器上部署最新的 [YOLOv8](https://github.com/ultralytics/ultralytics)、YOLOv8 Pose 和 [Nvidia TAO Models](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/cv_models/index.html)。我们还添加了 [ByteTrack](https://github.com/ifzhang/ByteTrack) 算法，以在低成本硬件上实现实时物体跟踪。

<div align="center"><img width="98%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-WebCam-Tracking.gif"/></div>

### Swift YOLO

我们实现了一个轻量级的目标检测算法，称为 Swift YOLO，它专为在计算能力有限的低成本硬件上运行而设计。可视化工具、模型训练和导出命令行界面现已重构。

<div align="center"><img width="98%" src="https://files.seeedstudio.com/sscma/docs/static/esp32/images/person_detection.png"/></div>

### 仪表识别

仪表是我们日常生活和工业生产中常见的仪器，例如模拟仪表、数字仪表等。SSCMA 提供了可以用来识别各种仪表读数的仪表识别算法。

<div align="center"><img width="98%" src="https://files.seeedstudio.com/sscma/docs/static/grove/images/pfld_meter.gif"/></div>

## 基准测试

SSCMA 旨在为嵌入式设备提供最佳性能和准确性，以下是最新算法的一些基准测试结果:

<div align="center"><img width="98%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-Swift-YOLO.png"/></div>

**注意: 基准测试主要包括 2 种架构，每种架构有 3 种不同大小 (输入尺寸 `[192, 224, 320]`，参数量可能有更多不同) 的模型，用图中点的大小表示。基准测试还包括量化模型，所有延迟都是在 NVIDIA A100上测量的。*

## SSCMA 工具链

SSCMA 提供了完整的工具链，让用户可以轻松地在低成本硬件上部署 AI 模型，包括：

- [SSCMA-Model-Zoo](https://github.com/Seeed-Studio/sscma-model-zoo) SSCMA 模型库为您提供了一系列针对不同应用场景的预训练模型。
- [SSCMA-Micro](https://github.com/Seeed-Studio/SSCMA-Micro) 一个跨平台的框架，用于在微控制器设备上部署和应用 SSCMA 模型。
- [Seeed-Arduino-SSCMA](https://github.com/Seeed-Studio/Seeed_Arduino_SSCMA) 支持 SSCMA-Micro 固件的 Arduino 库。
- [SSCMA-Web-Toolkit](https://seeed-studio.github.io/SenseCraft-Web-Toolkit) 一个基于 Web 的工具，用于更新设备固件、SSCMA 模型和参数。
- [Python-SSCMA](https://github.com/Seeed-Studio/python-sscma) 用于与微控制器进行交互的 Python 库，使用 SSCMA-Micro，并用于更高级别的深度学习应用。

## 致谢

SSCMA 是许多开发人员和贡献者的共同努力，感谢以下项目和组织对 SSCMA 的实现提供了参考和贡献:

- [OpenMMLab](https://openmmlab.com/)
- [ONNX](https://github.com/onnx/onnx)
- [NCNN](https://github.com/Tencent/ncnn)
- [TinyNN](https://github.com/alibaba/TinyNeuralNetwork)

## 许可证

本项目在 [Apache 2.0 开源许可证](LICENSE) 下发布。
