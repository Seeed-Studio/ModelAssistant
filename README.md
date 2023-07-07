# Seeed Studio EdgeLab

<div align="center">
  <img width="100%" src="docs/public/images/EdgeLab-Logo.png">
  <h3> <a href="https://seeed-studio.github.io/EdgeLab/"> Documentation </a> | <a href="https://github.com/Seeed-Studio/edgelab-model-zoo"> Model Zoo </a>  </h3>
</div>

English | [简体中文](README_zh-CN.md)


## Introduction

Seeed Studio EdgeLab is an open-source project focused on embedded AI. We have optimized excellent algorithms from [OpenMMLab](https://github.com/open-mmlab) for real-world scenarios and made implemention more user-friendly, achieving faster and more accurate inference on embedded devices.


## What's included

Currently we support the following directions of algorithms:

<details>
<summary>Anomaly Detection (coming soon)</summary>
In the real world, anomalous data is often difficult to identify, and even if it can be identified, it requires a very high cost. The anomaly detection algorithm collects normal data in a low-cost way, and anything outside normal data is considered anomalous. 
</details>

<details>
<summary>Computer Vision</summary>
Here we provide a number of computer vision algorithms such as object detection, image classification, image segmentation and pose estimation. However, these algorithms cannot run on low-cost hardware. EdgeLab optimizes these computer vision algorithms to achieve good running speed and accuracy in low-end devices.
</details>

<details>
<summary>Scenario Specific</summary>
EdgeLab provides customized scenarios for specific production environments, such as identification of analog instruments, traditional digital meters, and audio classification.
</details>

<br>

We will keep adding more algorithms in the future. Stay tuned!


## Features 

<details>
<summary>User-friendly</summary>
EdgeLab provides a user-friendly platform that allows users to easily perform training on collected data, and to better understand the performance of algorithms through visualizations generated during the training process. 
</details>

<details>
<summary>Models with low computing power and high performance</summary>
EdgeLab focuses on end-side AI algorithm research, and the algorithm models can be deployed on microprocessors, similar to <a href="https://www.espressif.com/en/products/socs/esp32">ESP32</a>, some <a href="https://arduino.cc">Arduino</a> development boards, and even in embedded SBCs such as <a href="https://www.raspberrypi.org">Raspberry Pi</a>.
</details>

<details>
<summary>Supports mutiple formats for model export</summary>
<a href="https://www.tensorflow.org/lite">TensorFlow Lite</a> is mainly used in microcontrollers, while <a href="https://onnx.ai">ONNX</a> is mainly used in devices with Embedded Linux. There are some special formats such as <a href="https://developer.nvidia.com/tensorrt">TensorRT</a>, <a href="https://docs.openvino.ai">OpenVINO</a> which are already well supported by OpenMMlab. EdgeLab has added TFLite model export for microcontrollers, which can be directly converted to uf2 format and drag-and-drop into the device for deployment.
</details>


## Acknowledgement

EdgeLab referenced the following projects:

- [OpenMMLab](https://openmmlab.com/)
- [ONNX](https://github.com/onnx/onnx)
- [NCNN](https://github.com/Tencent/ncnn)


## License

This project is released under the [MIT license](LICENSES).
