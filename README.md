# Seeed Studio Edgelab

## Introduction

This is a tool to develop audio and vision AI applications in a much easier and faster way using the OpenMMLab framework. 

By using OpenMMLab framework combined with [MMCV](https://github.com/open-mmlab/mmcv)(OpenMMLab foundational library for computer vision), you can easily develop a new backbone and use [MMClassification](https://github.com/open-mmlab/mmclassification), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMPose](https://github.com/open-mmlab/mmpose) to benchmark your backbone on image/ audio classification, object detection and pose estimation.

## Audio and Vision AI Tasks Supported

Currently we support the following tasks:

- Object detection 
- Meter reading using landmark detection
- Audio classification

We will keep adding more tasks in the future.

## Model Zoo

We also provide a model zoo for the tasks that are mentioned above

<table>
<thead>
  <tr>
    <th>Task</th>
    <th>Model Name</th>
    <th>Model Format</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="6">Audio classification</td>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/ali_classiyf_small_8k_35_8192.pth" target="_blank" rel="noopener noreferrer">ali_classiyf_small_8k_35_8192.pth</a></td>
    <td>pth</td>
  </tr>
  <tr>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/ali_classiyf_small_8k_35_8192.onnx" target="_blank" rel="noopener noreferrer">ali_classiyf_small_8k_35_8192.onnx</a></td>
    <td>onnx</td>
  </tr>
  <tr>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/ali_classiyf_small_8k_35_8192.zip" target="_blank" rel="noopener noreferrer">ali_classiyf_small_8k_35_8192.zip</a></td>
    <td>zip</td>
  </tr>
  <tr>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/ali_classiyf_small_8k_4_8192.pth" target="_blank" rel="noopener noreferrer">ali_classiyf_small_8k_4_8192.pth</a></td>
    <td>pth</td>
  </tr>
  <tr>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/ali_classiyf_small_8k_4_8192.onnx" target="_blank" rel="noopener noreferrer">ali_classiyf_small_8k_4_8192.onnx</a></td>
    <td>onnx</td>
  </tr>
  <tr>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/ali_classiyf_small_8k_4_8192.ncnn.zip" target="_blank" rel="noopener noreferrer">ali_classiyf_small_8k_4_8192.ncnn.zip</a></td>
    <td>zip</td>
  </tr>
  <tr>
    <td rowspan="3">Meter reading using <br>landmark detection</td>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/pfld_mv2n_112.pth" target="_blank" rel="noopener noreferrer">pfld_mv2n_112.pth</a></td>
    <td>pth</td>
  </tr>
  <tr>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/pfld_mv2n_112.onnx" target="_blank" rel="noopener noreferrer">pfld_mv2n_112.onnx</a></td>
    <td>onnx</td>
  </tr>
  <tr>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/pfld_mv2n_112.ncnn.zip" target="_blank" rel="noopener noreferrer">pfld_mv2n_112.ncnn.zip</a></td>
    <td>zip</td>
  </tr>
  <tr>
    <td rowspan="3">Object detection</td>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/yolov3_mbv2_416_coco.pth" target="_blank" rel="noopener noreferrer">yolov3_mbv2_416_coco.pth</a></td>
    <td>pth</td>
  </tr>
  <tr>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/yolov3_mbv2_416_coco.onnx" target="_blank" rel="noopener noreferrer">yolov3_mbv2_416_coco.onnx</a></td>
    <td>onnx</td>
  </tr>
  <tr>
    <td><a href="https://github.com/Seeed-Studio/Edgelab/releases/download/model_zoo/yolov3_mbv2_416_coco.ncnn.zip" target="_blank" rel="noopener noreferrer">yolov3_mbv2_416_coco.ncnn.zip</a></td>
    <td>zip</td>
  </tr>
</tbody>
</table>

## Meter Reading Detection in Action!

## Quick Getting Started 

The following is a quick getting started guide to train an audio classification model.

- First you will start with configuring the host environment 
- Then you need to determine the type of task you are doing, whether it is object detection, image/ audio classification
- After deciding, you can select the desired model according to your needs and determine the profile of the model
- Finally we export the trained PyTorch model to ONNX and NCNN formats  

### Prerequisites 

- PC with Ubuntu installed 
- Internet connection

### Configure host environment 

- **Step 1.** Clone the repo and access it

```
https://github.com/Seeed-Studio/edgelab
cd edgelab
```

- **Step 2.** Configure the host environment by running the following script which will download and install the relevant dependencies 

```sh
python3 tools/env_config.py
```

The above script will mainly install **PyTorch, MMCV, MMClassification, MMDetection and MMPose**.

**Note:** The above script will add various environment variables to the file ~/.bashrc, establish a conda virtual environment called edgelab, install relevant dependencies inside the virual environment. Eventhough everything is initialized, they are not activated.

- **Step 3.** Activate conda, virtual environments, and other related environment variables

```sh
source ~/.bashrc
conda activate edgelab
```

Now we have finished configuring the host environment 

### Configure the profile

Here we will choose the profile according to the task that we want to implement. We have prepared preconfigured files inside the the [configs](https://github.com/Seeed-Studio/edgelab/tree/master/configs) folder.

For example, we will choose an **audio classfication** example and use [ali_classiyf_small_8k_8192.py](https://github.com/Seeed-Studio/Edgelab/blob/master/configs/audio_classify/ali_classiyf_small_8k_8192.py) config file.

### Start training 

Execute the following command inside the activated conda virtual environment terminal to start training an end-to-end speech classification model.

```
python tools/train.py mmcls configs/audio_classify/ali_classiyf_small_8k_8192.py --gpus=0
```

The format of the above command looks like below

```
python tools/train.py <task_type> <config_file_location> --gpus=<cpu_or_gpu>
```

where:

- <task_type> refers to either **mmcls** for classfication, **mmdet** for detection and **mmpose** for pose estimation
- <config_file_location> refers to the path where the model configuration is located 
- <cpu_or_gpu> refers to specifying whether you want to train on CPU or GPU. Type **0** CPU and **1** for GPU

### Export to ONNX 

After the model training is completed, you can export the **.pth file** to the **ONNX file** format and convert it to other formats you want to use through ONNX. Assuming that the environment is in this project path, you can export the audio classfication model you just trained to the ONNX format by running the following command:

```sh
python tools/torch2onnx.py --config configs/audio_classify/ali_classiyf_small_8k_8192.py --checkpoint work_dirs/yolov3_192_node2_person/exp1/latest.pth --task mmcls --audio
```

The format of the above command looks like below

```
python tools/torch2onnx.py --config <config_file_location> --checkpoint <checkpoint_location> --task <task_type> --audio
```

where:

- <config_file_location> refers to the path where the model configuration is located 
- <checkpoint_location> refers to the .pth  model weight file generated after the training
- <task_type> refers to either **mmcls** for classfication, **mmdet** for detection and **mmpose** for pose estimation
- **--audio** parameter is added only when doing speech classfication

### Export to NCNN

Make sure you have already followed the previous step to conevert the .pth model file into ONNX 

Start the conversion process to NCNN by:

```sh
python tools/export_qiantize.py --onnx work_dirs/yolov3_192_node2_person/exp1 --type ncnn
```

The format of the above command looks like below

```sh
python tools/export_qiantize.py --onnx $ONNX_PATH --type $TYPE
```

where:

- **--$ONNX_PATH:** refers to the location of the weight file in ONNX format exported for the model before
- **--$TYPE:** refers to the optional parameters for what format you want to export the ONNX model to. You can choose from **onnx_fp16, onnx_quan_st, onnx_quan_dy, ncnn, ncnn_fp16, ncnn_quan**

## Further Guides 

- [Train an object detection model wth public datasets]()
- [Train an object detection model wth your own dataset]()
- [Train a meter reading detection model with your own dataset]()
- [Train an audio classfication model with your own dataset]()
- [Detailed guide on exporting ONNX to NCNN in quantized format]()