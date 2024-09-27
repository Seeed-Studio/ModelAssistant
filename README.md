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
    <a href="https://sensecraftma.seeed.cc"> Documentation </a> |
    <a href="https://sensecraftma.seeed.cc/introduction/installation"> Installation </a> |
    <a href="https://github.com/Seeed-Studio/ModelAssistant/tree/main/notebooks"> Colab </a> |
    <a href="https://github.com/Seeed-Studio/sscma-model-zoo"> Model Zoo </a> |
    <a href="https://seeed-studio.github.io/SenseCraft-Web-Toolkit"> Deploy </a> -
    <a href="README_zh-CN.md"> 简体中文 </a>
  </h3>

</div>

## Introduction

**S**eeed **S**ense**C**raft **M**odel **A**ssistant is an open-source project focused on providing state-of-the-art AI algorithms for embedded devices. It is designed to help developers and makers to easily deploy various AI models on low-cost hardwares, such as microcontrollers and single-board computers (SBCs).

<div align="center">

<img width="98%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-Deploy.gif"/>

</div>

**Real-world deploy examples on MCUs with less than 0.3 Watts power consumption.*

### 🤝 User-friendly

SSCMA provides a user-friendly platform that allows users to easily perform training on collected data, and to better understand the performance of algorithms through visualizations generated during the training process.

### 🔋 Models with low computing power and high performance

SSCMA focuses on end-side AI algorithm research, and the algorithm models can be deployed on microprocessors, similar to [ESP32](https://www.espressif.com.cn/en/products/socs/esp32), some [Arduino](https://arduino.cc) development boards, and even in embedded SBCs such as [Raspberry Pi](https://www.raspberrypi.org).

### 🗂️ Supports multiple formats for model export

[TensorFlow Lite](https://www.tensorflow.org/lite) is mainly used in microcontrollers, while [ONNX](https://onnx.ai) is mainly used in devices with Embedded Linux. There are some special formats such as [TensorRT](https://developer.nvidia.com/tensorrt), [OpenVINO](https://docs.openvino.ai) which are already well supported by OpenMMLab. SSCMA has added TFLite model export for microcontrollers, which can be directly converted to [TensorRT](https://developer.nvidia.com/tensorrt), [UF2](https://github.com/microsoft/uf2) format and drag-and-drop into the device for deployment.

## Features

We have optimized excellent algorithms from [OpenMMLab](https://github.com/open-mmlab) for real-world scenarios and made implementation more user-friendly, achieving faster and more accurate inference. Currently we support the following directions of algorithms:

### 🔍 Anomaly Detection

In the real world, anomalous data is often difficult to identify, and even if it can be identified, it requires a very high cost. The anomaly detection algorithm collects normal data in a low-cost way, and anything outside normal data is considered anomalous.

### 👁️ Computer Vision

Here we provide a number of computer vision algorithms such as **object detection, image classification, image segmentation and pose estimation**. However, these algorithms cannot run on low-cost hardwares. SSCMA optimizes these computer vision algorithms to achieve good running speed and accuracy in low-end devices.

### ⏱️ Scenario Specific

SSCMA provides customized scenarios for specific production environments, such as identification of analog instruments, traditional digital meters, and audio classification. We will continue to add more algorithms for specified scenarios in the future.

## What's New

SSCMA is always committed to providing the cutting-edge AI algorithms for best performance and accuracy, along with the community feedbacks, we keeps updating and optimizing the algorithms to meet the actual needs of users, here are some of the latest updates:

### 🔥 YOLO-World, MobileNetV4 and lighter SSCMA (Coming Soon)

We are working on the latest [YOLO-World](https://github.com/AILab-CVC/YOLO-World), [MobileNetV4](https://arxiv.org/abs/2404.10518) algorithms for embedded devices, we are also refactoring the SSCMA with less dependencies to make it more lightweight and easier to use, please stay tuned for the latest updates.

### YOLOv8, YOLOv8 Pose, Nvidia Tao Models and ByteTrack

With [SSCMA-Micro](https://github.com/Seeed-Studio/SSCMA-Micro), now you can deploy the latest [YOLOv8](https://github.com/ultralytics/ultralytics), YOLOv8 Pose, [Nvidia TAO Models](https://docs.nvidia.com/tao/tao-toolkit/text/model_zoo/cv_models/index.html) on microcontrollers. we also added the [ByteTrack](https://github.com/ifzhang/ByteTrack) algorithm to enable real-time object tracking on low-cost hardwares.

<div align="center"><img width="98%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-WebCam-Tracking.gif"/></div>

### Swift YOLO

We implemented a lightweight object detection algorithm called Swift YOLO, which is designed to run on low-cost hardware with limited computing power. The visualization tool, model training and export command-line interface has refactored now.

<div align="center"><img width="98%" src="https://files.seeedstudio.com/sscma/docs/static/esp32/images/person_detection.png"/></div>

### Meter Recognition

Meter is a common instrument in our daily life and industrial production, such as analog meters, digital meters, etc. SSCMA provides meter recognition algorithms that can be used to identify the readings of various meters.

<div align="center"><img width="98%" src="https://files.seeedstudio.com/sscma/docs/static/grove/images/pfld_meter.gif"/></div>

## Benchmarks

SSCMA aims to provide the best performance and accuracy for embedded devices, here are some benchmarks for the latest algorithms:

<div align="center"><img width="98%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-Swift-YOLO.png"/></div>

**Note: The bechmark mainly includes 2 architectures, each architecture has 3 models with different sizes (inputs `[192, 224, 320]`, parameters may various), represented by the size of the point in the graph. Quanitizied models are also included in the benchmark, all latency is measured on NVIDIA A100.*

## The SSCMA Toolchains

SSCMA provides a complete toolchain for users to easily deploy AI models on low-cost hardwares, including:

- [SSCMA-Model-Zoo](https://sensecraft.seeed.cc/ai/#/model) SSCMA Model Zoo provides a series of pre-trained models for different application scenarios for you to use. The source code for this web is [hosted here](https://github.com/Seeed-Studio/sscma-model-zoo).
- [SSCMA-Web-Toolkit, which is now renamed to SenseCraft AI](https://sensecraft.seeed.cc/ai/#/home) A web-based tool that makes trainning and deploying machine learning models (with a focus on vision models by now) fast, easy, and accessible to everyone. 
- [SSCMA-Micro](https://github.com/Seeed-Studio/SSCMA-Micro) A cross-platform framework that deploys and applies SSCMA models to microcontrol devices.
- [Seeed-Arduino-SSCMA](https://github.com/Seeed-Studio/Seeed_Arduino_SSCMA) Arduino library for devices supporting the SSCMA-Micro firmware.
- [Python-SSCMA](https://github.com/Seeed-Studio/python-sscma) A Python library for interacting with microcontrollers using SSCMA-Micro, and for higher-level deep learning applications.

## Acknowledgement

SSCMA is a united effort of many developers and contributors, we would like to thank the following projects and organizations for their contributions which SSCMA referenced to implement: 

- [OpenMMLab](https://openmmlab.com/)
- [ONNX](https://github.com/onnx/onnx)
- [NCNN](https://github.com/Tencent/ncnn)
- [TinyNN](https://github.com/alibaba/TinyNeuralNetwork)

## License

This project is released under the [Apache 2.0 license](LICENSE).
