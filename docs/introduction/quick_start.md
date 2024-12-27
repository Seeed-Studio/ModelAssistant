# Quick Start

In [Overview](./overview) we have introduced the functions and features provided by [SSCMA](https://github.com/Seeed-Studio/ModelAssistant). Considering that [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) is divided into multiple different modules, each module completing its corresponding tasks, we suggest following the steps below to quickly get started.

:::tip
We suggest that all beginners of [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) start learning from [Getting Started](#getting-started), if you are familiar with [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) or [OpenMMLab](https://github.com/open-mmlab), and you want to try to deploy on edge computing devices, modify existing neural networks, or train on user-defined data sets, you can directly refer to [Advanced](#advanced).
:::

## Getting Started

### Model Deployment

If you want to deploy the model on the device, please refer to the section [Deploy](../tutorials/deploy/overview) to learn how to deploy the model.

### Model Training

If you want to train a model, we highly recommend that you first try to train a model on the Colab platform. You can refer to the following tutorials:

#### Object Detection

| Model                                                                                           | Colab                                                                                                                                                                                                                     |
|:------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Gender_Detection_Swift-YOLO_192](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Gender_Detection_Swift-YOLO_192.md)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Gender_Detection_Swift-YOLO_192.ipynb)          |
| [Digital_Meter_Water_Swift-YOLO_192](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Digital_Meter_Water_Swift-YOLO_192.md)             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Digital_Meter_Water_Swift-YOLO_192.ipynb)       |
| [Apple_Detection_Swift-YOLO_192](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Apple_Detection_Swift-YOLO_192.md)                     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Apple_Detection_Swift-YOLO_192.ipynb)           |
| [person_Detection_Swift-YOLO_192](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/person_Detection_Swift-YOLO_192.md)                   | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/person_Detection_Swift-YOLO_192.ipynb)          |
| [Face_Detection_Swift-YOLO_96](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Face_Detection_Swift-YOLO_96.md)                         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Face_Detection_Swift-YOLO_96.ipynb)             |
| [COCO_Detection_Swift-YOLO_320](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/COCO_Detection_Swift-YOLO_320.md)                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/COCO_Detection_Swift-YOLO_320.ipynb)            |
| [Gesture_Detection_Swift-YOLO_192](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Gesture_Detection_Swift-YOLO_192.md)                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Gesture_Detection_Swift-YOLO_192.ipynb)         |
| [Digital_Meter_Electricity_Swift-YOLO_192](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Digital_Meter_Electricity_Swift-YOLO_192.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Digital_Meter_Electricity_Swift-YOLO_192.ipynb) |

#### Image Classification

| Model                                                                                                         | Colab                                                                                                                                                                                                                            |
|:--------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [MNIST_Classification_MobileNetV2_0.5_Rep_32](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/MNIST_Classification_MobileNetV2_0.5_Rep_32.md)         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/MNIST_Classification_MobileNetV2_0.5_Rep_32.ipynb)     |
| [Gender_Classification_MobileNetV2_0.35_Rep_64](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Gender_Classification_MobileNetV2_0.35_Rep_64.md)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Gender_Classification_MobileNetV2_0.35_Rep_64.ipynb)   |
| [Person_Classification_MobileNetV2_0.35_Rep_64](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Person_Classification_MobileNetV2_0.35_Rep_64.md)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Person_Classification_MobileNetV2_0.35_Rep_64.ipynb)   |
| [Person_Classification_MobileNetV2_0.35_Rep_96](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Person_Classification_MobileNetV2_0.35_Rep_96.md)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Person_Classification_MobileNetV2_0.35_Rep_96.ipynb)   |
| [Person_Classification_MobileNetV2_0.35_Rep_32](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/Person_Classification_MobileNetV2_0.35_Rep_32.md)     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/Person_Classification_MobileNetV2_0.35_Rep_32.ipynb)   |
| [CIFAR-10_Classification_MobileNetV2_0.35_Rep_32](https://github.com/seeed-studio/sscma-model-zoo/blob/main/docs/en/CIFAR-10_Classification_MobileNetV2_0.35_Rep_32.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seeed-studio/sscma-model-zoo/blob/main/notebooks/en/CIFAR-10_Classification_MobileNetV2_0.35_Rep_32.ipynb) |

## Advanced

1. First, refer to the [Installation Guide](./installation) to configure the running environment of [SSCMA](https://github.com/Seeed-Studio/ModelAssistant).

2. Then, familiar with the basic usage methods of [SSCMA](https://github.com/Seeed-Studio/ModelAssistant):

   - **Model Training**, please refer to [Model Training](../tutorials/training/overview) to learn how to use [SSCMA](https://github.com/Seeed-Studio/ModelAssistant) to train a model. We suggest that you select a model from an example for training.

   - **Model Export**. After completing model training, in order to deploy on the edge-computing device, it is necessary to first export the model.

   - **Model Verification**. Model verification can be performed after training or export. The former verifies the correctness of the neural network and training results, while the latter mainly verifies the correctness of the exported model, facilitating deployment and debugging on edge computing devices later. Some methods for model validation have been provided in the documents in the above two steps.

- **Model Deployment**. If you want to deploy the exported training model on edge computing devices, please refer to [ESP32 Deployment Example](../tutorials/deploy/xiao_esp32s3) or [Grove Vision AI Deployment Example](../tutorials/deploy/grove_vision_ai_v2).

- **Custom Datasets**. If you want to train on a custom dataset, please refer to [Datasets](../datasets/custom).

- **Custom Model**. If you want to modify an existing neural network or design your own neural network, please refer to [Model Configuration](../custom/basics).

## Necessary Knowledge

### 📸 Computer Vision:

The basics of computer vision are built upon digital image processing. So, you need to learn the basics of the DlP first. Then you can move forward to read computer vision topics like pattern recognition and 3D geometry. You need to know linear algebra to be able to fully understand some concepts of the computer vision like dimensionality reduction. After understanding the fundamentals of computer vision you should also build your knowledge in deep learning, especially in Convolutional Neural Networks (CNN).

### 💻 Programming:

Python will be enough for design and prototyping, but if you want to do some
embedded work, you should also be familiar with C++.

### 🧰 Tools:

OpenCV is the main tool for computer vision, and Numpy is an important tool for data processing and analysis. You must know them. You never know, but you should know what tools are available and how to use them. How to use them. Another tool you need to familiarize yourself with is the deep learning framework. Frameworks. You can start with Keras which is the easiest to learn and then learn Tensorflow or PyTorch.
