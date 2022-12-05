# Train an object detection model with your own dataset

**Note:** Before proceeding, please follow the steps under **Configure host environment** inside the [README.md](https://github.com/Seeed-Studio/Edgelab/blob/master/README.md)

### Configure the profile

Here we will choose the profile according to the task that we want to implement. We have prepared preconfigured files inside the the [configs](https://github.com/Seeed-Studio/edgelab/tree/master/configs) folder.

For our object detection example, we will use [yolov3_mbv2_416_coco.py](https://github.com/Seeed-Studio/Edgelab/blob/master/configs/yolo/yolov3_mbv2_416_coco.py) config file. This file will be used to configure the dataset for training including the dataset location, number of classes and the label names.

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

- **Step 16.** Go to **Downloads** folder and unzip the downloaded file

```sh
cd ~/Downloads
unzip **file_name.zip**

## for example
unzip Apple\ Detection.v1i.coco.zip
```

- **Step 17.** Open the file **yolov3_mbv2_416_coco.py** which is available inside **~/edgelab/configs/yolo** directory using any text editor and change the following

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