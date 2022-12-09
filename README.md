# Seeed Studio Edgelab

![EdgeLab-logo](https://user-images.githubusercontent.com/20147381/206450696-66aca04f-81a7-40c7-aa31-79f8b7b2b522.png)

## Introduction

Seeed Studio Edgelab is an open-source project focused on embedded AI. We have optimized excellent algorithms from [OpenMMLab](https://github.com/open-mmlab) for real-world scenarios and made implemention more user-friendly, achieving faster and more accurate inference on embedded devices.

OpenMMLab currently has 15 algorithm libraries covering many directions such as classification, object detection, segmentation, pose estimation, etc. There are more than 250 algorithm implementations, but like most algorithms on the market today, not many of these algorithms can be implemented in real products or limited to high-performance GPUs. Based on OpenMMLab, EdgeLab selects algorithms that meet the hardware performance and optimizes the algorithm implementation according to the required  scenarios, so that the algorithms can run faster, with less power consumption and more accurately in the devices. 

## What's included?

Currently we support the following directions of algorithms:

<details>
<summary>Anomaly Detection (coming soon)</summary>
In the real world, anomalous data is often difficult to identify, and even if it can be identified, it requires a very high cost. The anomaly detection algorithm collects normal data in a low-cost way, and anything outside normal data is considered anomalous. 
</details>

<details>
<summary>Object Detection</summary>
YOLO-based object detection algorithms have achieved more than 0.75 on the COCO dataset for mAP. However, these object detection algorithms cannot run on low-cost hardware. EdgeLab optimizes YOLO algorithms to achieve good running speed and accuracy in low-end devices.
</details>

<details>
<summary>Discrete Classification</summary>
Except for sound and visual, most of the data in the real world are discrete, and the data can only produce results after classification.
</details>

<details>
<summary>Scenario-specific</summary>
Specific scenarios, such as the recognition of analog meters, or traditional digital meters.
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
EdgeLab focuses on end-side AI algorithm research, and the algorithm models can be deployed on microprocessors, similar to ESP32, some Arduino development boards, and even in embedded SBCs such as Raspberry Pi.
</details>

<details>
<summary>Supports mutiple formats for model export</summary> 
At present, TensorFlow Lite is mainly used in microcontrollers, while ONNX is mainly used in devices with embedded Linux. There are some special formats such as TensorRT, OpenVINO, which are already well supported by OpenMMlab. EdgeLab has added TFLite model export for microcontrollers, which can be directly converted to uf2 format and drag-and-drop into the device for deployment.
</details>

## Explore more guides!

<details>
<summary>Click here</summary>

- [Train an object detection model with public datasets](https://github.com/Seeed-Studio/Edgelab/blob/master/docs/Object-detection-public-dataset.md)
- [Train an object detection model with your own dataset](https://github.com/Seeed-Studio/Edgelab/blob/master/docs/Object-detection-own-dataset.md)
- [Train a meter reading detection model with existing dataset](https://github.com/Seeed-Studio/Edgelab/blob/master/docs/Meter-reading-detection-existing-dataset.md)
</details>
