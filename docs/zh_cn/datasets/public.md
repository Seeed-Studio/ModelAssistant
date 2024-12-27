# 公共数据集

SSCMA 支持多种公共数据集，包括互联网上公开的 COCO (Common Objects in Context) 数据集、ImageNet 数据集、由 Seeed 制作的 FOMO、Meter 等数据集等。


## 获取公共数据集

您也可以从其它平台如 Roboflow、Kaggle 等下载数据集，只要这些数据集符合 SSCMA 支持的数据集格式即可。

### SSCMA

[SSCMA](https://github.com/Seeed-Studio/ModelAssistant) 目前提供以下官方数据集，用于对应模型的训练和测试。

对于使用命令下载的数据集，请确保在运行命令前处于**项目根目录**，命令会自动下载数据集并将其保存在当前目录下的名为 `datasets` 的文件夹中，并完成解压。

- [下载指针 Meter 数据集](https://files.seeedstudio.com/sscma/datasets/meter.zip):

  ```sh
  wget https://files.seeedstudio.com/sscma/datasets/meter.zip -P datasets -O datasets/meter.zip && \
  mkdir -p datasets/meter && \
  unzip datasets/meter.zip -d datasets
  ```

- [下载口罩 COCO 数据集](https://files.seeedstudio.com/sscma/datasets/coco_mask.zip):

  ```sh
  wget https://files.seeedstudio.com/sscma/datasets/coco_mask.zip -P datasets -O datasets/coco_mask.zip && \
  mkdir -p datasets/coco_mask && \
  unzip datasets/coco_mask.zip -d datasets/coco_mask
  ```

### Roboflow

[Roboflow](https://public.roboflow.com/) 是公共计算机视觉数据集的免费托管平台，支持的格式包括包括 CreateML JSON、COCO JSON、Pascal VOC XML、YOLO 和 Tensorflow TFRecords 等，还额外添加了对应数据集的缩小和增强版本。

:::tip

我们十分推荐您在这里寻找数据集，您只需要注册一个账号，就可以免费下载数百个来自互联网的不同数据集，用于满足您的特定需求。

:::

你可以在 Roboflow 上找到一些 SSCMA 可用的的数据集，如下所示:

| Dataset | Description |
| -- | -- |
| [Digital Meter Water](https://universe.roboflow.com/seeed-studio-dbk14/digital-meter-water/dataset/1) | Digital Meter Water Dataset |
| [Digital Meter Seg7](https://universe.roboflow.com/seeed-studio-dbk14/digital-meter-seg7/dataset/1) | Digital Meter Seg7 Dataset |
| [Digit Seg7 Classification](https://universe.roboflow.com/seeed-studio-ovcjn/digit-seg7/1) | Digit Seg7 Classification Dataset |

### Kaggle

[Kaggle](https://www.kaggle.com/) 是一个数据建模和数据分析竞赛平台。企业和研究者可在其上发布数据，统计学者和数据挖掘专家可在其上进行竞赛以产生最好的模型。Kaggle 也提供了数以千计的数据集，您可以访问 [Kaggle 数据集](https://www.kaggle.com/datasets) 挑选适合您需求的数据集。

### 其它数据集平台

您可以参考 [Datasets List](https://www.datasetlist.com/) 网站，该网站提供了大量的公共数据集，您可以根据自己的需求选择合适的数据集，需要注意的是，这些数据集可能需要经过转换才能在 SSCMA 中使用。 


## 常用公开数据集

我们在此列举了一些常用的公共数据集，您可以根据自己的需求选择合适的数据集。

### COCO 数据集

[COCO 数据集](https://cocodataset.org/) 是一个大规模的目标检测、分割和描述数据集，包含 330K 图像、1.5 百万目标实例、80 个目标类别、91 个材质类别、每张图 5 个描述和 25 万个带关键点的人像。

#### 应用场景

- **计算机视觉研究**：用于开发和测试图像识别、目标检测、图像分割和字幕生成等算法。
- **自动驾驶**：用于训练车辆识别和定位道路上的行人、车辆和其他物体。
- **机器人视觉**：用于训练机器人识别和操作环境中的物体。
- **安防监控**：用于检测监控视频中的异常行为或特定对象。
- **医疗影像分析**：用于辅助诊断，通过图像分割技术识别病变区域。
- **增强现实（AR）**：用于在现实世界中叠加虚拟信息，需要精确的目标检测和图像分割技术。
- **电子商务**：用于商品图像的自动分类和检索。

### ImageNet 数据集

[ImageNet 数据集](http://www.image-net.org/) 是一个大型的视觉对象识别数据库，包含超过 1400 万张经过标注的图像，涵盖了超过 2 万个类别。

#### 应用场景

- **计算机视觉研究**：用于开发和测试图像识别和目标定位算法。
- **深度学习模型训练**：用于训练大规模的深度学习模型，如卷积神经网络（CNN）。
- **图像搜索引擎**：用于提高图像搜索的准确性和相关性。
- **安防监控**：用于识别和定位监控视频中的特定对象。
- **自动驾驶**：用于训练车辆识别道路中的行人、车辆等。
- **智能监控**：用于识别和分类监控摄像头捕获的图像。
- **生物信息学**：用于分析和分类生物图像，如细胞和组织。
- **内容过滤和版权保护**：用于自动识别和过滤不适当的内容或保护版权图像。

### FOMO 数据集

[FOMO 数据集](https://files.seeedstudio.com/sscma/datasets/coco_mask.zip) 是由 Seeed 制作的一个包含人脸和口罩的数据集，用于训练和测试口罩检测模型，其与 COCO 数据集有相同的目录结构和标注格式。

#### 应用场景

- **疫情防控**：用于检测人员是否佩戴口罩，确保公共场所的安全。
- **安防监控**：用于监控视频中的人员佩戴情况，及时发现异常情况。
- **智能门禁**：用于识别人员是否佩戴口罩，控制门禁系统的开关。
- **智能巡检**：用于检测工人是否佩戴口罩，确保工作场所的安全。
- **智能安检**：用于检测乘客是否佩戴口罩，确保交通工具的安全。

### Meter 数据集

[Meter 数据集](https://files.seeedstudio.com/sscma/datasets/meter.zip) 是由 Seeed 制作的一个包含指针表计和数字表计的数据集，用于训练和测试表计识别模型，其与 YOLO TXT 数据集有相同的目录结构和标注格式。

#### 应用场景

- **智能仪表读数**：用于自动识别和读取仪表上的指针读数。
- **智能计量**：用于自动计量仪表上的指针读数，提高计量的准确性。
- **智能监控**：用于监控仪表的读数，及时发现异常情况。


