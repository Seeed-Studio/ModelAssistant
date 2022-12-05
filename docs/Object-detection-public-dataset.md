# Train an object detection model with public datasets

**Note:** Before proceeding, please follow the steps under **Configure host environment** inside the [README.md](https://github.com/Seeed-Studio/Edgelab/blob/master/README.md)

### Configure the profile

Here we will choose the profile according to the task that we want to implement. We have prepared preconfigured files inside the the [configs](https://github.com/Seeed-Studio/edgelab/tree/master/configs) folder.

For our object detection example, we will use [yolov3_mbv2_416_coco.py](https://github.com/Seeed-Studio/Edgelab/blob/master/configs/yolo/yolov3_mbv2_416_coco.py) config file. This file will be used to configure the dataset for training including the dataset location, number of classes and the label names.

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

If you navigate to **~/edgelab/work_dirs/yolov3_mbv2_416_coco.py/exp1** folder, you will see the trained PyTorch model file as **latest.pt**

<div align=center><img width=1000 src="https://files.seeedstudio.com/wiki/Edgelab/4.jpg"/></div>