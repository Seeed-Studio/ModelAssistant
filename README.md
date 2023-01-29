# Seeed Studio EdgeLab


<div align="center">
  <img width="100%" src="https://user-images.githubusercontent.com/20147381/206665275-feceede2-c68c-4259-a4db-541b3bd25b2f.png">
  <h3> <a href="https://edgelab.readthedocs.io/en/latest/"> Documentation </a> | <a href="https://edgelab.readthedocs.io/zh_CN/latest/"> 中文文档 </a>  </h3>
</div>

[![Documentation Status](https://readthedocs.org/projects/edgelab/badge/?version=latest)](https://edgelab.readthedocs.io/en/latest/?badge=latest)

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
<summary>Scenario-Specific</summary>
Specific scenarios, such as the recognition of analog meters, traditional digital meters and audio classfication.
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
<a href="https://www.tensorflow.org/lite">TensorFlow Lite</a>
is mainly used in microcontrollers, while <a href="https://onnx.ai">ONNX</a> is mainly used in devices with Embedded Linux. There are some special formats such as <a href="https://developer.nvidia.com/tensorrt">TensorRT</a>, <a href="https://docs.openvino.ai">OpenVINO</a> which are already well supported by OpenMMlab. EdgeLab has added TFLite model export for microcontrollers, which can be directly converted to uf2 format and drag-and-drop into the device for deployment.
</details>

## Experience EdgeLab in 3 easy steps!

Now let's experience EdgeLab in the fastest way by deploying it on [Grove - Vision AI Module](https://www.seeedstudio.com/Grove-Vision-AI-Module-p-5457.html) and [SenseCAP A1101](https://www.seeedstudio.com/SenseCAP-A1101-LoRaWAN-Vision-AI-Sensor-p-5367.html)!

<details>
<summary>1. Download pretrained model and firmware</summary>

- **Step 1.** We provide 2 different models for object detection and analog meter reading detection. Click on the model that you want to use to download it.

    - [Analog meter reading detection model](https://files.seeedstudio.com/wiki/EdgeLab/uf2/analog-meter-model.uf2)
    - Object detection model (coming soon!)
    
- **Step 2.** We provide 2 different firmware for Grove - Vision AI and SenseCAP A1101. Click on the firmware that you want to use to download it.

    - Analog meter reading detection
    
        - [Grove - Vision AI](https://files.seeedstudio.com/wiki/EdgeLab/uf2/grove-vision-ai-firmware.uf2)
        - [SenseCAP A1101](https://files.seeedstudio.com/wiki/EdgeLab/uf2/sensecap-A1101-firmware.uf2)
    - Object detection

        - Coming soon!
    
</details>

<details>
<summary>2. Deploy model and firmware</summary>

- **Step 1.** Connect Grove - Vision AI Module/ SenseCAP A1101 to PC by using USB Type-C cable 

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/45.png"/></div>

- **Step 2.** Double click the boot button to enter **boot mode**

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/46.png"/></div>

- **Step 3:** After this you will see a new storage drive shown on your file explorer as **GROVEAI** for **Grove - Vision AI Module** and as **VISIONAI** for **SenseCAP A1101**

<div align=center><img width=500 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/62.jpg"/></div>

- **Step 4:** Drag and drop the previous **firmware.uf2** at first, and then the **model.uf2** file to **GROVEAI** or **VISIONAI** 

Once the copying is finished **GROVEAI** or **VISIONAI** drive will disapper. This is how we can check whether the copying is successful or not.
</details>

<details>
<summary>3. View live detection results</summary>

- **Step 1:** After loading the firmware and connecting to PC, visit [this URL](https://files.seeedstudio.com/grove_ai_vision/index.html)

- **Step 2:** Click **Connect** button. Then you will see a pop up on the browser. Select **Grove AI - Paired** and click **Connect**
  
<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/13.jpg"/></div>

<div align=center><img width=400 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/12.png"/></div>

Upon successful connection, you will see a live preview from the camera. Here the camera is pointed at an analog meter.

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/14.png"/></div>

Now we need to set 3 points which is the center point, start point and the end point. 

- **Step 3:** Click on **Set Center Point** and click on the center of the meter. you will see a pop up confirm the location and press **OK**

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/15.png"/></div>

You will see the center point is already recorded

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/16.png"/></div>

- **Step 4:** Click on **Set Start Point** and click on the first indicator point. you will see a pop up confirm the location and press **OK**

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/17.png"/></div>

You will see the first indicator point is already recorded

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/18.png"/></div>

- **Step 5:** Click on **Set End Point** and click on the last indicator point. you will see a pop up confirm the location and press **OK**

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/19.png"/></div>

You will see the last indicator point is already recorded

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/20.png"/></div>

- **Step 6:** Set the measuring range according to the first digit and last digit of the meter. For example, he we set as **From:0 To 0.16**

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/21.png"/></div>

- **Step 7:** Set the number of decimal places that you want the result to display. Here we set as 2

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/22.png"/></div>

Finally you can see the live meter reading results as follows

<div align=center><img width=800 src="https://files.seeedstudio.com/wiki/EdgeLab/meter-own-github/meter.gif"/></div>
</details>

## Getting Started with EdgeLab

We provide end-to-end getting started guides for EdgeLab where you can prepare datasets, train and finally deploy AI models to embedded edge AI devices such as the Grove - Vision AI Module and SenseCAP A1101 for different AI applications. 

Here we introduce 2 different platforms to run the commands. 

- Linux PC with a powerful GPU 
- Google Colab workspace 

The advantage of using Google Colab is that you can use any device having a web browser. In addition, it already comes with high performance GPUs for training. Use the below links to access the tutorials. 

- [Getting Started with EdgeLab on Local PC](docs/EdgeLab-getting-started.md)
- Getting Started with EdgeLab on Google Colab

    - [Analog meter reading detection with existing dataset](docs/Analog_meter_detection_existing_dataset.ipynb) 
    - [Analog meter reading detection with own dataset](docs/Analog_meter_detection_own_dataset.ipynb)