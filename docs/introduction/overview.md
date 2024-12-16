<div align="center">
  <img width="20%" src="https://files.seeedstudio.com/sscma/docs/images/SSCMA-Hero.png" />
</div>

# Project Introduction

Seeed SenseCraft Model Assistant (SSCMA) is an open-source project focused on embedded artificial intelligence. We optimize for real-world scenarios and are committed to providing a great user experience, enabling faster and more accurate inference on embedded devices.

## What's Included?

Currently, we support algorithms in the following directions:

### 🔍 Anomaly Detection

In the real world, anomalous data is often difficult to identify and can be very costly even when identified. Anomaly detection algorithms collect normal data at a low cost and consider anything outside of normal data as anomalous.

### 👁️ Computer Vision

We provide a variety of computer vision algorithms such as object detection, image classification, image segmentation, and pose estimation. However, these algorithms cannot run on low-cost hardware. [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) optimizes these computer vision algorithms to achieve good speed and accuracy on low-end devices.

### ⏱️ Specific Scenarios

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) offers customized scenarios for specific production environments, such as simulation instrument recognition, traditional digital instrument, and audio classification. We will continue to add more algorithms for specified scenarios in the future.

## Features

### 🤝 User-Friendly

We provide a user-friendly platform that allows users to easily train collected data and better understand the performance of algorithms through the visualization results generated during the training process.

### 🔋 High-Performance Models with Low Computational Power

We focus on edge AI algorithm research, and our algorithm models can be deployed on microprocessors, such as [ESP32](https://www.espressif.com/en/products/socs/esp32), some [Arduino](https://arduino.cc) development boards, and even embedded SBCs like [Raspberry Pi](https://www.raspberrypi.org).

### 🗂️ Support for Multiple Model Export Formats

We are committed to addressing the fragmentation of models deployed on the edge, supporting multiple model export formats. For example, [TensorFlow Lite](https://www.tensorflow.org/lite) is mainly used for microcontrollers, while [ONNX](https://onnx.ai) is more common on embedded Linux devices. There are also special formats closely related to hardware, such as [TensorRT](https://developer.nvidia.com/tensorrt), [OpenVINO](https://docs.openvino.ai), [Vela](https://developer.arm.com/documentation/109267/latest/Tool-support-for-the-Arm-Ethos-U-NPU/Ethos-U-Vela-compiler), [HailoRT](https://hailo.ai/products/hailo-software/hailo-ai-software-suite/#sw-hailort), which are well-supported.

### 🚀 Future-Oriented

As artificial intelligence moves towards the edge, balancing low latency, low power consumption, and low cost against high precision and efficiency has always been a challenge, but it is an inevitable trend for the future. Therefore, many special hardware accelerators (NPU, TPU, VPU, LPU, etc.) have emerged, but these hardware often have limited support for models or have special requirements for the structure and operators of a model. In practical application scenarios, the same model may need to be deployed on different hardware, which requires not only model conversion but also partial structural modifications. This requires starting from the deployment end, defining algorithms in a well-supported form, training and tuning on a large amount of data, forming a closed loop, and truly realizing the landing of edge AI. We will continue to focus on this field and provide better support for users through technologies such as QAT, MLIR, TVM, etc.

## SSCMA Toolchain

SSCMA provides a complete toolchain that allows users to easily deploy AI models on low-cost hardware, including:

- [SSCMA-Model-Zoo](https://github.com/Seeed-Studio/sscma-model-zoo) provides a series of pre-trained models for different application scenarios.
- [SSCMA-Micro](https://github.com/Seeed-Studio/SSCMA-Micro) is a cross-platform framework for deploying and applying SSCMA models on microcontroller devices.
- [Seeed-Arduino-SSCMA](https://github.com/Seeed-Studio/Seeed_Arduino_SSCMA) is an Arduino library that supports SSCMA-Micro firmware.
- [SSCMA-Web-Toolkit](https://seeed-studio.github.io/SenseCraft-Web-Toolkit) is a web-based tool for updating device firmware, SSCMA models, and parameters.
- [Python-SSCMA](https://github.com/Seeed-Studio/python-sscma) is a Python library for interacting with microcontrollers, using SSCMA-Micro, and for higher-level deep learning applications.

## Application Examples

SSCMA can be applied in many fields to solve various practical problems, empower, and improve production efficiency. Here are some examples:

### Object Detection

Object detection is a computer vision technology aimed at identifying and locating specific objects in images or videos. It can not only recognize the categories of objects present in the image (such as people, vehicles, animals, etc.) but also determine the location of these objects in the image, usually represented by bounding boxes.

<div align="center"><img width="800" src="https://files.seeedstudio.com/sscma/docs/static/esp32/images/person_detection.png"/></div>

### Analog Meter Reading

Using artificial intelligence technology to identify and read values on analog meters (such as electric meters, water meters, etc.).

<div align="center"><img width="800" src="https://files.seeedstudio.com/sscma/docs/static/grove/images/pfld_meter.gif"/></div>

### Digital Meter Reading

Similar to analog meters, using artificial intelligence technology to identify and read values on digital meters (such as electronic meters, digital electric meters, etc.).

<div align="center"><img width="800" src="https://files.seeedstudio.com/sscma/docs/static/grove/images/digital_meter.gif"/></div>

For more applications, please refer to [Model Zoo](https://github.com/Seeed-Studio/sscma-model-zoo).
