# Seeed Studio EdgeLab

<div align="center">
  <img width="100%" src="docs/public/images/EdgeLab-Logo.png">
  <h3> <a href="https://seeed-studio.github.io/EdgeLab/zh_cn/"> 中文文档 </a> | <a href="https://github.com/Seeed-Studio/edgelab-model-zoo"> Model Zoo </a>  </h3>
</div>

[English](README.md) | 简体中文

## 简介

Seeed Studio EdgeLab 是一个专注于嵌入式人工智能的开源项目。我们对 [OpenMMLab](https://github.com/open-mmlab) 的优秀算法进行了优化，使其适用于现实世界的场景，并使实施更加人性化，在嵌入式设备上实现更快和更准确的推理。

## 包括什么

目前我们支持以下的算法方向:

<details>
<summary>异常检测</summary>
在现实世界中，异常数据往往很难被识别，即使能被识别，也需要很高的成本。异常检测算法以低成本的方式收集正常数据，任何超出正常数据的东西都被认为是异常的。
</details>

<details>
<summary>计算机视觉</summary>
这里我们提供了一些计算机视觉算法，如物体检测、图像分类、图像分割和姿态估计。然而，这些算法不能在低成本的硬件上运行。EdgeLab 对这些计算机视觉算法进行了优化，以便在低端设备中实现良好的运行速度和准确性。
</details>

<details>
<summary>特定场景</summary>
EdgeLab 为特定的生产环境提供定制场景的解决方案，例如模拟仪表、传统数字仪表的读数和音频分类。
</details>

<br>

我们将在未来不断增加更多的算法。敬请关注!

## 特点介绍

<details>
<summary>用户友好</summary>
EdgeLab 提供了一个用户友好的平台，使用户能够轻松地对收集的数据进行训练，并通过训练过程中产生的可视化效果更好地了解算法的性能。
</details>

<details>
<summary>高性能模型，低算力开销</summary>
EdgeLab 专注于终端人工智能算法研究，算法模型可以部署在微处理器上，类似于 <a href="https://www.espressif.com/en/products/socs/esp32">ESP32</a>，一些 <a href="https://arduino.cc">Arduino</a> 开发板，甚至是嵌入式 SBC，如 <a href="https://www.raspberrypi.org">Raspberry Pi</a>
</details>

<details>
<summary>多模型格式导出</summary>
<a href="https://www.tensorflow.org/lite">TensorFlow Lite</a> 主要用于微控制器，而 <a href="https://onnx.ai">ONNX</a> 主要用于嵌入式 Linux 的设备。有一些特殊的格式，如 <a href="https://developer.nvidia.com/tensorrt">TensorRT</a>、<a href="https://docs.openvino.ai">OpenVINO</a>，已经被 OpenMMlab 很好地支持.
</details>

## 致谢

EdgeLab 参考了以下项目:

- [OpenMMLab](https://openmmlab.com)
- [ONNX](https://github.com/onnx/onnx)
- [NCNN](https://github.com/Tencent/ncnn)

## 开源许可证

该项目采用 [MIT 开源许可证](LICENSES)。
