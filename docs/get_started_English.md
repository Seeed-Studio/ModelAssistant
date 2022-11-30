# Seeed Studio Edgelab

## Introduction

This is a tool to develop speech and vision AI applications in a much easier and faster way using the OpenMMLab framework. 

With OpenMMLab framework, you can easily develop a new backbone and use MMClassification, MMDetection and MMPose to benchmark your backbone on image/ audio classification, object detection and pose estimation tasks.

## Getting started 

- First you will start with configuring the host environment 
- Then you need to determine the type of task you are doing, whether it is object detection, image/ audio classification
- After deciding, you can select the desired model according to your needs and determine the profile of the model
- Here we demonstrate how to train an object detection model using the COCO dataset
- We also demonstrate how to use other public datasets and your own dataset to train an object detection model
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

The above script will mainly install **PyTorch** and the following **OpenMMLab** third-party libraries


- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark. Besides classification, it's also a repository to store various backbones
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab Pose Estimation Toolbox and Benchmark

**Note:** The above script will add various environment variables to the file ~/.bashrc, establish a conda virtual environment called edgelab, install relevant dependencies inside the virual environment. Eventhough everything is initialized, they are not activated.

- **Step 3.** Activate conda, virtual environments, and other related environment variables

```sh
source ~/.bashrc
conda activate edgelab
```

Now we have finished configuring the host environment 

### Configure the profile

Here we will choose the profile according to the task that we want to implement. We have prepared preconfigured files inside the the [configs](https://github.com/Seeed-Studio/edgelab/tree/master/configs) folder.

For our object detection example, we will use [yolov3_mbv2_416_coco.py](https://github.com/Seeed-Studio/edgelab/blob/master/configs/yolo/yolov3_mbv2_416_coco.py) config file. This file will be used to configure the dataset for training including the dataset location, number of classes and the label names.

#### Use COCO dataset

Here we will first use the COCO dataset which is a public dataset. We have already configured **yolov3_mbv2_416_coco.py** file out-of-the-box to download and include the COCO dataset. So you do not need to do any further configuration here.

#### Use other public datasets

You can also use other public datasets available online. Simply search **open-source datasets** on Google and choose from a variety of datasets available. Make sure to download the dataset in **COCO** or **Pascal VOC** format because only those 2 formats are supported by training at the moment. 

Here we will demonstrate how to use [Roboflow Universe](https://universe.roboflow.com) to download a public dataset ready for training. Roboflow Universe is a recommended platform which provides a wide-range of datasets and it has [90,000+ datasets with 66+ million images](https://blog.roboflow.com/computer-vision-datasets-and-apis) available for building computer vision models.

- **Step 1.** Visit [this URL](https://universe.roboflow.com/lakshantha-dissanayake/apple-detection-5z37o/dataset/1) to access an Apple Detection dataset available publicly on Roboflow Universe. This is a dataset with a single class named as **apple**

- **Step 2.** Click **Create Account** to create a Roboflow account

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/Edgelab/1.png"/></div>

- **Step 3.** Click **Download**, select **COCO** as the **Format**, click **download zip to computer** and click **Continue**

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/Edgelab/2.jpg"/></div>

**Note:** You can also choose **Pascal VOC** as the **Format**

This will place a **.zip file** in the **Downloads** folder on your PC

- **Step 4.** Go to **Downloads** folder and unzip the downloaded file

```sh
cd ~/Downloads
unzip **file_name.zip**

## for example
unzip Apple\ Detection.v1i.coco.zip
```

- **Step 5.** Open the file **yolov3_mbv2_416_coco.py** which is available inside **~/edgelab/configs/yolo** directory using any text editor and change the following

1. num_classes=<number_of_classes>
2. data_root = '<root_location_of_dataset>'
3. classes=('<class_name>',)
4. ann_file='<location_of_annotation_file_corresponding_to_train/valid/test>'
5. img_prefix='<location_of_dataset_corresponding_to_train/valid/test>',

**Note:** You will see **<----Change** in places where you want to update the code

```sh
..................
..................
# model settings
model = dict(
    ..................
    ..................
    bbox_head=dict(type='YOLOV3Head',
                   num_classes=1, <----Change
    ..................
    ..................
# dataset settings
dataset_type = 'CustomCocoDataset'
data_root = '/home/<username>/Downloads/' <----Change
..................
..................
data = dict(
    ..................
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(type=dataset_type,
                     data_root=data_root,
                     classes=('apple',), <----Change
                     ann_file='/home/<username>/Downloads/train/_annotations.coco.json', <----Change
                     img_prefix='/home/<username>/Downloads/train', <----Change
                     pipeline=train_pipeline)),
    val=dict(type=dataset_type,
             data_root=data_root,
             classes=('apple',), <----Change
             ann_file='/home/<username>/Downloads/valid/_annotations.coco.json', <----Change
             img_prefix='/home/<username>/Downloads/valid', <----Change
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              data_root=data_root,
              classes=('apple',), <----Change
              ann_file='/home/<username>/Downloads/test/_annotations.coco.json', <----Change
              img_prefix='/home/<username>/Downloads/test', <----Change
              pipeline=test_pipeline))
..................
..................
```

#### Use your own dataset

If you use your own dataset, you will need to annotate all the images in your dataset. Annotating means simply drawing rectangular boxes around each object that we want to detect and assign them labels. We will explain how to do this using Roboflow.

Here we will use a dataset with images containing apples

- **Step 1.** Click [here](https://app.roboflow.com/login) to sign up for a Roboflow account

- **Step 2.** Click **Create New Project** to start our project

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/YOLOV5/2.jpg"/></div>

- **Step 3.** Fill in **Project Name**, keep the **License (CC BY 4.0)** and **Project type (Object Detection (Bounding Box))**  as default. Under **What will your model predict?** column, fill in an annotation group name. For example, in our case we choose **apples**. This name should highlight all of the classes of your dataset. Finally, click **Create Public Project**.

<div align=center><img width=350 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/6.jpg"/></div>

- **Step 4.** Drag and drop the images that you have captured

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/7.png"/></div>

- **Step 5.** After the images are processed, click **Finish Uploading**. Wait patiently until the images are uploaded.

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/4.jpg"/></div>

- **Step 6.** After the images are uploaded, click **Assign Images**

<div align=center><img width=300 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/5.jpg"/></div>

- **Step 7.** Select an image, draw a rectangular box around an apple, choose the label as **apple** and press **ENTER**

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/9.png"/></div>

- **Step 8.** Repeat the same for the remaining apples

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/10.png"/></div>

**Note:** Try to label all the apples that you see inside the image. If only a part of an apple is visible, try to label that too.

- **Step 9.** Continue to annotate all the images in the dataset

Roboflow has a feature called **Label Assist** where it can predict the labels beforehand so that your labelling will be much faster. However, it will not work with all object types, but rather a selected type of objects. To turn this feature on, you simply need to press the **Label Assist** button, **select a model**, **select the classes** and navigate through the images to see the predicted labels with bounding boxes

<div align=center><img width=200 src="https://files.seeedstudio.com/wiki/YOLOV5/41.png"/></div>

<div align=center><img width=400 src="https://files.seeedstudio.com/wiki/YOLOV5/39.png"/></div>

<div align=center><img width=400 src="https://files.seeedstudio.com/wiki/YOLOV5/40.png"/></div>

As you can see above, it can only help to predict annotations for the 80 classes mentioned. If your images do not contain the object classes from above, you cannot use the label assist feature.

- **Step 10.** Once labelling is done, click **Add images to Dataset**

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/YOLOV5/25.jpg"/></div>

- **Step 11.** Next we will split the images between "Train, Valid and Test". Keep the default percentages for the distribution and click **Add Images**

<div align=center><img width=330 src="https://files.seeedstudio.com/wiki/YOLOV5/26.png"/></div>

- **Step 12.** Click **Generate New Version**

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/YOLOV5/27.jpg"/></div>

- **Step 13.** Now you can add **Preprocessing** and **Augmentation** if you prefer

- **Step 14.** Next, proceed with the remaining defaults and click **Generate**

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/SenseCAP-A1101/14.png"/></div>

- **Step 15.** Click **Download**, select **COCO** as the **Format**, click **download zip to computer** and click **Continue**

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/Edgelab/2.jpg"/></div>

**Note:** You can also choose **Pascal VOC** as the **Format**

This will place a **.zip file** in the **Downloads** folder on your PC

- **Step 16.** Repeat **steps 4 and 5** of the previous section with the title **Use other public datasets**

### Start training 

Execute the following command inside the activated conda virtual environment terminal to start training an end-to-end object detection model.

```sh
python tools/train.py mmdet configs/yolo/yolov3_mbv2_416_coco.py --gpus=0 --cfg-options runner.max_epochs=100
```

The format of the above command looks like below

```sh
python tools/train.py <task_type> <config_file_location> --gpus=<cpu_or_gpu> --cfg-options runner.max_epochs=<number_of_epochs>
```

where:

- <task_type> refers to either **mmcls** for classfication, **mmdet** for detection and **mmpose** for pose estimation
- <config_file_location> refers to the path where the model configuration is located 
- <cpu_or_gpu> refers to specifying whether you want to train on CPU or GPU. Type **0** CPU and **1** for GPU
- --cfg-options runner.max_epochs=<number_of_epochs> refers to the number of training cycles

**Note:** If your GPU gives an error saying that there is not enough GPU memory, you can pass the below paramaters to the above command

```sh
--cfg-options data.samples_per_gpu=8

## if there is still error with the above
--cfg-options data.samples_per_gpu=4
```

After the model training is completed, you will see the below output 

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/Edgelab/3.png"/></div>

If you navigate to **~/edgelab/work_dirs/yolov3_mbv2_416_coco/exp1** folder, you will see the trained PyTorch model file as **best.pt**

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/Edgelab/4.jpg"/></div>

### Export to ONNX 

After the model training is completed, you can export the **.pth file** to the **ONNX file** format and convert it to other formats you want to use through ONNX. Assuming that the environment is in this project path, you can export the object detection model you just trained to the ONNX format by running the following command:

```sh
python tools/torch2onnx.py --config configs/yolo/yolov3_mbv2_416_coco.py --checkpoint work_dirs/yolov3_mbv2_416_coco/exp1/latest.pth --task mmdet
```

The format of the above command looks like below

```
python tools/torch2onnx.py --config <config_file_location> --checkpoint <checkpoint_location> --task <task_type>
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
python tools/export_qiantize.py --onnx work_dirs/yolov3_mbv2_416_coco/exp1 --type ncnn
```

The format of the above command looks like below

```sh
python tools/export_qiantize.py --onnx $ONNX_PATH --type $TYPE
```

where:

- **--$ONNX_PATH:** refers to the location of the weight file in ONNX format exported for the model before
- **--$TYPE:** refers to the optional parameters for what format you want to export the ONNX model to. You can choose from **onnx_fp16, onnx_quan_st, onnx_quan_dy, ncnn, ncnn_fp16, ncnn_quan**