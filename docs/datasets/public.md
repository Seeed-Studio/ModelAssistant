# Public Datasets

SSCMA supports a variety of public datasets, including the publicly available COCO (Common Objects in Context) dataset, ImageNet dataset, and datasets such as FOMO and Meter created by Seeed, among others.

## Obtaining Public Datasets

You can also download datasets from other platforms such as Roboflow, Kaggle, etc., as long as these datasets conform to the formats supported by SSCMA.

### SSCMA

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) currently offers the following official datasets for training and testing corresponding models.

For datasets downloaded using commands, please ensure you are in the **project root directory** before running the commands. The commands will automatically download the datasets and save them in a folder named `datasets` in the current directory, and complete the unzipping process.

- [Download Meter Dataset](https://files.seeedstudio.com/sscma/datasets/meter.zip):

  ```sh
  wget https://files.seeedstudio.com/sscma/datasets/meter.zip  -P datasets -O datasets/meter.zip && \
  mkdir -p datasets/meter && \
  unzip datasets/meter.zip -d datasets
  ```

- [Download Mask COCO Dataset](https://files.seeedstudio.com/sscma/datasets/coco_mask.zip):

  ```sh
  wget https://files.seeedstudio.com/sscma/datasets/coco_mask.zip  -P datasets -O datasets/coco_mask.zip && \
  mkdir -p datasets/coco_mask && \
  unzip datasets/coco_mask.zip -d datasets/coco_mask
  ```

### Roboflow

[Roboflow](https://public.roboflow.com/) is a free hosting platform for public computer vision datasets, supporting formats including CreateML JSON, COCO JSON, Pascal VOC XML, YOLO, and Tensorflow TFRecords, among others, with additional缩小 and augmentation versions for corresponding datasets.

:::tip

We highly recommend looking for datasets here. You only need to register an account to download hundreds of datasets from the internet for free, to meet your specific needs.

:::

You can find some datasets available for SSCMA on Roboflow, as shown below:

| Dataset | Description |
| -- | -- |
| [Digital Meter Water](https://universe.roboflow.com/seeed-studio-dbk14/digital-meter-water/dataset/1)  | Digital Meter Water Dataset |
| [Digital Meter Seg7](https://universe.roboflow.com/seeed-studio-dbk14/digital-meter-seg7/dataset/1)  | Digital Meter Seg7 Dataset |
| [Digit Seg7 Classification](https://universe.roboflow.com/seeed-studio-ovcjn/digit-seg7/1)  | Digit Seg7 Classification Dataset |

### Kaggle

[Kaggle](https://www.kaggle.com/) is a data modeling and data analysis competition platform. Companies and researchers can publish data on it, and statisticians and data mining experts can compete to produce the best models. Kaggle also offers thousands of datasets, and you can visit [Kaggle Datasets](https://www.kaggle.com/datasets) to select datasets that meet your needs.

### Other Dataset Platforms

You can refer to the [Datasets List](https://www.datasetlist.com/) website, which provides a large number of public datasets. You can choose the appropriate datasets according to your needs. It is important to note that these datasets may need to be converted before they can be used in SSCMA.

## Common Public Datasets

We list some common public datasets here, and you can choose the appropriate datasets according to your needs.

### COCO Dataset

[COCO Dataset](https://cocodataset.org/) is a large-scale object detection, segmentation, and captioning dataset, containing 330K images, 1.5 million object instances, 80 object categories, 91 material categories, 5 captions per image, and 250,000 images with keypoints.

#### Application Scenarios

- **Computer Vision Research**: Used for developing and testing image recognition, object detection, image segmentation, and caption generation algorithms.
- **Autonomous Driving**: Used for training vehicles to recognize and locate pedestrians, vehicles, and other objects on the road.
- **Robot Vision**: Used for training robots to recognize and manipulate objects in the environment.
- **Security Surveillance**: Used for detecting abnormal behaviors or specific objects in surveillance videos.
- **Medical Image Analysis**: Used for auxiliary diagnosis, identifying lesion areas through image segmentation techniques.
- **Augmented Reality (AR)**: Used for overlaying virtual information on the real world, requiring precise object detection and image segmentation techniques.
- **E-commerce**: Used for automatic classification and retrieval of product images.

### ImageNet Dataset

[ImageNet Dataset](http://www.image-net.org/) is a large visual object recognition database containing over 14 million annotated images across more than 20,000 categories.

#### Application Scenarios

- **Computer Vision Research**: Used for developing and testing image recognition and object localization algorithms.
- **Deep Learning Model Training**: Used for training large-scale deep learning models, such as Convolutional Neural Networks (CNNs).
- **Image Search Engines**: Used to improve the accuracy and relevance of image searches.
- **Security Surveillance**: Used for identifying and locating specific objects in surveillance videos.
- **Autonomous Driving**: Used for training vehicles to recognize pedestrians, vehicles, etc., on the road.
- **Intelligent Monitoring**: Used for recognizing and classifying images captured by surveillance cameras.
- **Bioinformatics**: Used for analyzing and classifying biological images, such as cells and tissues.
- **Content Filtering and Copyright Protection**: Used for automatically recognizing and filtering inappropriate content or protecting copyrighted images.

### FOMO Dataset

[FOMO Dataset](https://files.seeedstudio.com/sscma/datasets/coco_mask.zip) is a dataset containing faces and masks created by Seeed, used for training and testing mask detection models, with the same directory structure and annotation format as the COCO dataset.

#### Application Scenarios

- **Epidemic Prevention and Control**: Used to detect whether people are wearing masks to ensure safety in public places.
- **Security Surveillance**: Used to monitor the wearing situation of people in surveillance videos and detect abnormalities in a timely manner.
- **Intelligent Access Control**: Used to recognize whether people are wearing masks to control the opening and closing of access control systems.
- **Intelligent Patrol**: Used to detect whether workers are wearing masks to ensure safety in the workplace.
- **Intelligent Security Check**: Used to detect whether passengers are wearing masks to ensure the safety of transportation tools.

### Meter Dataset

[Meter Dataset](https://files.seeedstudio.com/sscma/datasets/meter.zip) is a dataset containing analog and digital meter data created by Seeed, used for training and testing meter recognition models, with the same directory structure and annotation format as the YOLO TXT dataset.

#### Application Scenarios

- **Intelligent Meter Reading**: Used for automatically recognizing and reading pointer readings on meters.
- **Intelligent Metering**: Used for automatically metering pointer readings on meters to improve the accuracy of metering.
- **Intelligent Monitoring**: Used for monitoring the readings of meters to detect abnormalities in a timely manner.
