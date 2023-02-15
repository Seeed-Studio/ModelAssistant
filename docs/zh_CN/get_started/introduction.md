# 项目简介

![EdgeLab-logo-2](././_static/EdgeLab-logo.png)

## 简介

Seeed Studio EdgeLab是一个专注于嵌入式人工智能的开源项目。我们对[OpenMMLab](https://github.com/open-mmlab)的优秀算法进行了优化，使其适用于现实世界的场景，并使实施更加人性化，在嵌入式设备上实现更快、更准确的推理。

## 包括什么

目前，我们支持以下方向的算法。

<details>
<summary>异常检测（即将推出）</summary>。
在现实世界中，异常数据往往很难被识别，即使能被识别，也需要很高的成本。异常检测算法以低成本的方式收集正常数据，任何超出正常数据的东西都被认为是异常的。
</details>

<details></details>
<summary>计算机视觉</summary>
这里我们提供了一些计算机视觉算法，如物体检测、图像分类、图像分割和姿态估计。然而，这些算法不能在低成本的硬件上运行。EdgeLab对这些计算机视觉算法进行了优化，以便在低端设备中实现良好的运行速度和准确性。
</details>

<details>
<summary>场景定制</summary>
对于具体生产环境进行专业的场景定制化方案，如模拟仪表、传统数字仪表和音频分类的识别。
</details>

<br>

我们将在未来不断增加更多的算法。敬请关注!

## Features 

<details>
<summary>用户友好</summary>
EdgeLab提供了一个用户友好的平台，使用户能够轻松地对收集的数据进行训练，并通过训练过程中产生的可视化效果更好地了解算法的性能。
</details>

<details>
<summary>低算力平台支持</summary>
EdgeLab专注于终端侧的人工智能算法研究，算法模型可以部署在微处理器上， 例如<a href="https://www.espressif.com/en/products/socs/esp32">ESP32</a>, 一些 <a href="https://arduino.cc">Arduino</a> 开发板, 甚至在嵌入式SBC，如 <a href="https://www.raspberrypi.org">Raspberry Pi</a>.
</details>

<details>
<summary>多格式导出</summary>
<a href="https://www.tensorflow.org/lite">TensorFlow Lite</a>在嵌入式设备中广泛应用,  <a href="https://onnx.ai">ONNX</a>则被用在嵌入式Linux中流行. 有一些特殊的格式，如<a href="https://developer.nvidia.com/tensorrt">TensorRT</a>、<a href="https://docs.openvino.ai">OpenVINO</a>，这些格式已经被OpenMMlab很好地支持。EdgeLab为微控制器增加了TFLite模型导出，可以直接转换为uf2格式，并拖放到设备中进行部署。
</details>


