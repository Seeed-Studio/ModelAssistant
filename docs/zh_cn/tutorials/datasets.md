# 数据集

EdgeLab 支持多种数据集。您可以在互联网上浏览并下载不同的数据集，或是自行标注、制作数据集。

## 互联网数据集

### EdgeLab

EdgeLab 目前提供以下官方数据集，用于对应模型的训练和测试。

对于使用命令下载的数据集，请确保在运行命令前处于 **EdgeLab 项目根目录**，命令会自动下载数据集并将其保存在当前目录下的名为 `datasets` 的文件夹中，并完成解压。

- [下载自定义 Meter 数据集](https://files.seeedstudio.com/wiki/Edgelab/meter.zip):

  ```sh
  wget https://files.seeedstudio.com/wiki/Edgelab/meter.zip -P datasets && unzip datasets/meter.zip -d datasets
  ```

- [下载 COCO_MASK 数据集](https://files.seeedstudio.com/wiki/Edgelab/coco_mask.zip):

  ```sh
  wget https://files.seeedstudio.com/wiki/Edgelab/coco_mask.zip -P datasets && unzip datasets/coco_mask.zip -d datasets
  ```

### Roboflow

[Roboflow](https://public.roboflow.com/) 是公共计算机视觉数据集的免费托管平台，支持的格式包括包括 CreateML JSON、COCO JSON、Pascal VOC XML、YOLO 和 Tensorflow TFRecords 等，还额外添加了对应数据集的缩小和增强版本。

:::tip

我们十分推荐您在这里寻找数据集，您只需要注册一个账号，就可以免费下载数百个来自互联网的不同数据集，用于满足您的特定需求。

:::

### Kaggle

[Kaggle](https://www.kaggle.com/) 是一个数据建模和数据分析竞赛平台。企业和研究者可在其上发布数据，统计学者和数据挖掘专家可在其上进行竞赛以产生最好的模型。Kaggle 也提供了数以千计的数据集，您可以访问 [Kaggle 数据集](https://www.kaggle.com/datasets) 挑选适合您需求的数据集。

## 自定义数据集

创建自定义数据集通常包括以下步骤:

1. **收集数据:** 收集与问题域相关的数据。这些数据可以是文本、图像、音频或视频等格式。

2. **整理数据:** 对收集的数据进行清洗、[标注](#%E6%95%B0%E6%8D%AE%E9%9B%86%E6%A0%87%E6%B3%A8)、去重等操作，以确保数据的准确性和一致性。这一步骤是确保训练出的模型准确性的关键。

3. **划分数据集:** 将整理好的数据集划分成训练集、验证集和测试集。通常采用 70%、15%、15% 的比例划分数据集。

4. **转换数据格式:** 将整理好的数据集转换成模型可以读取的格式，如文本格式、图像格式等。

5. **加载数据集:** 将转换好的数据集加载到模型中进行训练和测试。在加载数据集时需要注意的是，要使用合适的数据加载器和批量大小。

6. **数据增强 (可选，建议由 EdgeLab 完成):** 对数据集进行数据增强，如旋转、翻转、剪裁等操作，以增加数据集的多样性和数量。

## 数据集标注

标注数据集是将数据集中的样本进行分类或者打上标签的过程，通常需要进行人工干预。

标注数据集的过程是非常关键的，它决定了训练出的模型的质量。下面是标注数据集的一些常见方式和工具:

- **手动标注:** 通过手工对数据集进行标注的方式，对每个样本进行标注，可以确保标注的准确性，但是速度较慢。

- **半自动标注:** 将人工标注的结果应用到其他数据集中，减少标注时间，但标注的准确性可能有所降低。

- **自动标注:** 使用一些算法模型对数据进行自动标注，例如关键字提取、文本分类等。虽然可以提高标注效率，但标注的准确性也可能会受到影响。

常用的数据标注工具包括:

- [LabelImg](https://github.com/heartexlabs/labelImg): 适用于图像标注的工具，支持多种标注格式，如 PASCAL VOC、YOLO 等。

- [Labelbox](https://labelbox.com/): 一个在线标注工具，支持图像、文本、视频等格式的标注，具有多种标注模板和自定义标注模板功能。

- [Doccano](https://github.com/doccano/doccano): 一款用于文本分类和序列标注的开源标注工具，支持多种标注格式，如 NER、POS 等。

- [Annotator](https://github.com/openannotation/annotator): 一个轻量级的在线标注工具，支持图像、文本、音频等格式的标注。

- [VGG Image Annotator (VIA)](https://gitlab.com/vgg/via): 一个用于图像标注的开源工具，支持多种标注格式，如 PASCAL VOC、YOLO 等。

- [COCO Annotator](https://github.com/jsbroks/coco-annotator): 一个基于 Web 的图像和视频注释工具，可用于目标检测、分割、关键点标注等任务。

以上是一些常见的数据标注工具，不同的工具适用于不同的数据集类型和标注需求，可以根据实际需求进行选择。
