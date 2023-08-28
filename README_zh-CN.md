# SenseCraft Model Assistant by Seeed Studio

<div align="center">
  <img width="100%" src="docs/public/images/SSCMA-Logo.png">
  <h3><a href="https://sensecraftma.seeed.cc/sscma">文档</a> | <a href="https://github.com/Seeed-Studio/sscma-model-zoo">模型仓库</a></h3>
</div>

英文 | [简体中文](README_zh-CN.md)

## 简介

SSCMA 是一个专注于嵌入式人工智能的开源项目。我们从 [OpenMMLab](https://github.com/open-mmlab) 优化了优秀的算法，并使实现更加用户友好，在嵌入式设备上实现更快速、更准确的推理。


## 包含内容

目前，我们支持以下算法方向：

### 🔍 异常检测
在现实世界中，异常数据通常难以识别，即使能够识别出来，也需要很高的成本。异常检测算法以低成本的方式收集正常数据，认为任何超出正常数据范围的数据都是异常的。

### 👁️ 计算机视觉
我们提供了许多计算机视觉算法，例如目标检测、图像分类、图像分割和姿态估计。但是，这些算法无法在低成本硬件上运行。SenseCraft 模型助手优化了这些计算机视觉算法，实现了较好的运行速度和准确性。

### ⏱️ 场景特定
SenseCraft 模型助手为特定的生产环境提供了定制化场景，例如模拟仪器、传统数字仪表和音频分类的识别。我们将继续在未来添加更多的指定场景算法。

## 特点

### 🤝 用户友好
SenseCraft 模型助手提供了一个用户友好的平台，方便用户使用收集的数据进行训练，并通过训练过程中生成的可视化结果更好地了解算法的性能。

### 🔋 低计算功耗、高性能的模型
SenseCraft 模型助手专注于边缘端人工智能算法研究，算法模型可以部署在微处理器上，类似于 [ESP32] (https://www.espressif.com.cn/en/products/socs/esp32)、一些 [Arduino](https://arduino.cc) 开发板，甚至在嵌入式 SBCs（如 [Raspberry Pi](https://www.raspberrypi.org) ）上。

### 🗂️ 支持多种模型导出格式
[TensorFlow Lite](https://www.tensorflow.org/lite) 主要用于微控制器，而 [ONNX](https://onnx.ai) 主要用于嵌入式Linux设备。还有一些特殊格式，如 [TensorRT](https://developer.nvidia.com/tensorrt)、[OpenVINO](https://docs.openvino.ai)，这些格式已经得到 OpenMMLab 的良好支持。SenseCraft 模型助手添加了 TFLite 模型导出功能，可直接转换为 [TensorRT](https://developer.nvidia.com/tensorrt) 和 [UF2](https://github.com/microsoft/uf2) 格式，并可拖放到设备上进行部署。

## 应用示例

### 目标检测
<div align=center><img width=800 src="./docs/public/static/esp32/images/person_detection.png"/></div>

### 模拟仪器识别
<div align=center><img width=800 src="./docs/public/static/grove/images/pfld_meter.gif"/></div>

### 传统数字仪表识别
<div align=center><img width=800 src="./docs/public/static/grove/images/digital_meter.gif"/></div>

更多应用示例请参考 [模型仓库](https://github.com/Seeed-Studio/sscma-model-zoo)。

## 致谢

SenseCraft模型助手参考了以下项目：

- [OpenMMLab](https://openmmlab.com/)
- [ONNX](https://github.com/onnx/onnx)
- [NCNN](https://github.com/Tencent/ncnn)
- [TinyNN](https://github.com/alibaba/TinyNeuralNetwork)

## 许可证

本项目在[MIT许可证](LICENSES)下发布。
