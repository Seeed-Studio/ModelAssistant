## 数据集格式

SSCMA 支持多种数据集格式，包括 COCO、YOLO 等，您可以根据自己的需求选择合适的数据集格式，下面我们将介绍我们支持的数据集格式。

### COCO 格式

COCO 数据集的目录结构通常如下：

```
COCO/
  annotations/
    instances_val2017.json
    instances_train2017.json
    instances_val2014.json
    instances_train2014.json
    ...
  img/
    train2017/
      image1.jpg
      image2.jpg
      ...
    val2017/
      image1.jpg
      image2.jpg
      ...
    train2014/
      image1.jpg
      image2.jpg
      ...
    val2014/
      image1.jpg
      image2.jpg
      ...
```

- `annotations/`：包含所有标注文件，通常是 JSON 格式。
- `img/`：包含实际的图像文件，分为训练集和验证集，以及不同的年份版本。

COCO 数据集的标注文件是 JSON 格式的，其结构如下：

```json
{
  "info": {},
  "licenses": [],
  "images": [
    {
      "id": int,
      "width": int,
      "height": int,
      "file_name": str,
      "license": int,
      "flickr_url": str,
      "coco_url": str,
      "date_captured": str,
      "tags": list
    }
    ...
  ],
  "annotations": [
    {
      "id": int,
      "image_id": int,
      "category_id": int,
      "segmentation": list,
      "area": float,
      "bbox": [x, y, width, height],
      "iscrowd": 0 or 1
    }
    ...
  ],
  "categories": [
    {
      "id": int,
      "name": str,
      "supercategory": str
    }
    ...
  ]
}
```

- `info`：包含数据集的信息。
- `licenses`：包含使用许可的信息。
- `images`：包含图像的信息，如 ID、尺寸、文件名等。
- `annotations`：包含标注信息，如对象的 ID、图像 ID、类别 ID、分割信息、面积、边界框等。
- `categories`：包含类别的信息，如 ID 和名称。

在 SSCMA 中，FOMO 数据集的标注文件是 COCO 格式的 JSON 文件，您可以直接使用 COCO 数据集的标注文件进行训练和测试。

### YOLO 格式

SSCMA 的指针 Meter 数据集使用了类似 YOLO 格式数据集的结构，其目录如下：

```
Meter/
  data_root/
    images/          # 存放图像文件
    *.jpg            # 图像文件
    annotations/     # 存放标注文件
    *.json or *.txt  # 标注文件，可以是 JSON 或 TXT 格式
```

对于，JSON 格式，其标注结构如下：

```json
{
  "imagePath": "path/to/image.jpg",
  "shapes": [
    {
      "points": [[x1, y1], [x2, y2], ...]  // 关键点坐标
    }
  ]
}
```

- `imagePath`：图像文件的路径。
- `shapes`：包含一个或多个形状的列表，每个形状由关键点坐标组成。

对于 TXT 格式，其标注结构如下：

```
image1.jpg x1 y1 x2 y2 ... xn yn
image2.jpg x1 y1 x2 y2 ... xn yn
...
```

- 每行代表一个图像的标注信息，第一列为图像文件名，后续列为关键点坐标。

### ImageNet 格式

ImageNet 数据集的目录结构通常如下：

```
ImageNet/
  train/
    n01440764/
      image1.jpg
      image2.jpg
      ...
    n01443537/
      image1.jpg
      image2.jpg
      ...
    ...
  val/
    n01440764/
      image1.jpg
      image2.jpg
      ...
    n01443537/
      image1.jpg
      image2.jpg
      ...
    ...
```

- `train/`：包含训练集图像。
- `val/`：包含验证集图像。
- `test/`：包含测试集图像。

每个类别的图像都被放在以类别 ID 命名的子目录中。

ImageNet 数据集的标注通常以纯文本文件的形式存在，每个类别一个文件。标注文件的结构如下：

```
n01440764/image1.jpg 0
n01440764/image1.jpg 1 
...
```

- 每一行代表一个图像中的一个标注，第一列为图像文件路径，第二列为类别 ID。

### Anomaly 格式

Anomaly 数据集的目录结构如下：

```
Anomaly/
  data_root/
    anomaly/          # 存放异常音频文件
    *.npy             # 异常音频数据文件
    normal/           # 存放正常音频文件
    *.npy             # 正常音频数据文件
```

- `data_root/`：数据集的根目录。
- `anomaly/`：包含所有标记为异常的音频文件。
- `normal/`：包含所有标记为正常的音频文件。
- `*.npy`：音频数据文件，以NumPy数组格式存储。

默认情况下，我们可以仅使用正常的音频文件进行训练，无需对数据集进行标注。

关于数据集类型，分为以下两种情况:

对于 Signal 数据集，每个 `.npy` 文件都包含了一组 N 轴陀螺仪和加速度计的 X 条数据，其形状为 `(N, W, H)`， 其中 N 是对应传感器每条轴，每个 Channel 下，由 W，H 组成一个二维数组，表示 X 条传感器数据，W 和 H 由小波变换和马尔科夫链确定，每个数据的值为 0-255 的整数。

对于 Microphone 数据集，每个 `.npy` 文件都包含了一组 N 个音频数据，其形状为 `(N, W, H)`，其中 N 是音频数据的 Channel 数，单个 Microphone 情况下为 1，W 和 H 由梅尔频谱变换确定，每个数据的值为 0-255 的整数。

## 数据集扩展

您可以根据自己的需求使用拓展的数据加载器件。

### LanceDB

LanceDB 是一个基于 PyArrow 的数据存储和处理库，它提供了高效的列式存储和处理能力，适用于大规模数据集的存储和访问。在本教程中，我们将介绍如何使用 LanceDB 扩展数据集，使 SSCMA 能够更高效地存储和访问数据。


#### 数据存储结构和方式

根据提供的代码，数据库中的数据应该以以下结构和方式存储：

1. **表结构**：数据被存储在表中，每个表由多个列组成，每列对应一个字段（如 `image`, `filename`, `bbox`, `catid`, `label`）。

2. **字段类型**：
   - `image`：二进制数据（`pa.binary()`），存储图像的字节数据。
   - `filename`：字符串类型（`pa.string()`），存储图像文件的名称。
   - `bbox`：二进制数据（`pa.binary()`），存储边界框信息的字节数据。
   - `catid`：二进制数据（`pa.binary()`），存储类别ID信息的字节数据。
   - `label`：整型（`pa.int8()`），存储图像的标签或类别。

3. **数据编码**：边界框（`bbox`）和类别 ID（`catid`）数据被编码为字节数据并存储在二进制列中。

4. **数据批处理**：数据通过 `process_images_detect` 和 `process_images_cls` 函数以批处理的方式生成 PyArrow 的 `RecordBatch` 对象，然后写入 LanceDB。

5. **数据读取**：使用 `LanceDataset` 类从 LanceDB 读取数据，其中 `load_data_list` 方法从数据库中读取所有数据，`get_data_info` 方法获取单个数据项的详细信息。

6. **数据解码**：在 `get_data_info` 方法中，将存储为字节数据的 `bbox` 和 `catid` 字段解码回 NumPy 数组，以便进一步处理。

通过这种方式，数据集在 LanceDB 中的存储结构清晰，且便于高效读取和处理。