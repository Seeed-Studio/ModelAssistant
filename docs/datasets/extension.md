## Dataset Formats

SSCMA supports a variety of dataset formats, including COCO, YOLO, and more. You can choose the dataset format that best suits your needs, and below we will introduce the dataset formats we support.

### COCO Format

The directory structure of a COCO dataset typically looks like this:

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

- `annotations/`: Contains all annotation files, usually in JSON format.
- `img/`: Contains the actual image files, divided into training and validation sets, as well as different year versions.

The annotation files of the COCO dataset are in JSON format, and their structure is as follows:

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

- `info`: Contains information about the dataset.
- `licenses`: Contains information about usage licenses.
- `images`: Contains image information, such as ID, dimensions, filename, etc.
- `annotations`: Contains annotation information, such as object ID, image ID, category ID, segmentation information, area, bounding box, etc.
- `categories`: Contains category information, such as ID and name.

In SSCMA, the annotation files for the FOMO dataset are JSON files in COCO format, and you can directly use the annotation files of the COCO dataset for training and testing.

### YOLO Format

SSCMA's pointer Meter dataset uses a structure similar to the YOLO format dataset, and its directory is as follows:

```
Meter/
  data_root/
    images/          # Stores image files
    *.jpg            # Image files
    annotations/     # Stores annotation files
    *.json or *.txt  # Annotation files, can be in JSON or TXT format
```

For JSON format, the annotation structure is as follows:

```json
{
  "imagePath": "path/to/image.jpg",
  "shapes": [
    {
      "points": [[x1, y1], [x2, y2], ...]  // Keypoint coordinates
    }
  ]
}
```

- `imagePath`: The path to the image file.
- `shapes`: Contains a list of one or more shapes, each composed of keypoint coordinates.

For TXT format, the annotation structure is as follows:

```
image1.jpg x1 y1 x2 y2 ... xn yn
image2.jpg x1 y1 x2 y2 ... xn yn
...
```

- Each line represents the annotation information for an image, with the first column being the image filename and the subsequent columns being the keypoint coordinates.

### ImageNet Format

The directory structure of the ImageNet dataset typically looks like this:

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

- `train/`: Contains training set images.
- `val/`: Contains validation set images.
- `test/`: Contains test set images.

Images for each category are placed in a subdirectory named after the category ID.

The annotations for the ImageNet dataset usually exist in the form of plain text files, one file per category. The structure of the annotation file is as follows:

```
n01440764/image1.jpg 0
n01440764/image1.jpg 1
...
```

- Each line represents an annotation for an image, with the first column being the image file path and the second column being the category ID.

### Anomaly Format

The directory structure of the Anomaly dataset is as follows:

```
Anomaly/
  data_root/
    anomaly/          # Stores abnormal audio files
    *.npy             # Abnormal audio data files
    normal/           # Stores normal audio files
    *.npy             # Normal audio data files
```

- `data_root/`: The root directory of the dataset.
- `anomaly/`: Contains all audio files marked as abnormal.
- `normal/`: Contains all audio files marked as normal.
- `*.npy`: Audio data files stored in NumPy array format.

By default, we can use only the normal audio files for training without annotating the dataset.

Regarding dataset types, there are the following two situations:

For Signal datasets, each `.npy` file contains a set of N-axis gyroscope and accelerometer data with X entries, shaped as `(N, W, H)`, where N corresponds to each sensor axis, and each Channel is composed of a two-dimensional array of W and H, representing X sensor data entries. W and H are determined by wavelet transformation and Markov chains, and each data value is an integer between 0-255.

For Microphone datasets, each `.npy` file contains a set of N audio data entries, shaped as `(N, W, H)`, where N is the number of audio data Channels, which is 1 for a single Microphone, and W and H are determined by the Mel spectrogram transformation, with each data value being an integer between 0-255.

## Dataset Expansion

You can use expanded data loading components according to your needs.

### LanceDB

LanceDB is a data storage and processing library based on PyArrow, providing efficient columnar storage and processing capabilities suitable for the storage and access of large-scale datasets. In this tutorial, we will introduce how to use LanceDB to expand the dataset, enabling SSCMA to store and access data more efficiently.

#### Data Storage Structure and Method

According to the provided code, data in the database should be stored with the following structure and method:

1. **Table Structure**: Data is stored in tables, each consisting of multiple columns, with each column corresponding to a field (e.g., `image`, `filename`, `bbox`, `catid`, `label`).

2. **Field Types**:
   - `image`: Binary data (`pa.binary()`), storing the byte data of the image.
   - `filename`: String type (`pa.string()`), storing the name of the image file.
   - `bbox`: Binary data (`pa.binary()`), storing the byte data of the bounding box information.
   - `catid`: Binary data (`pa.binary()`), storing the byte data of the category ID information.
   - `label`: Integer type (`pa.int8()`), storing the label or category of the image.

3. **Data Encoding**: Bounding box (`bbox`) and category ID (`catid`) data are encoded as byte data and stored in binary columns.

4. **Data Batch Processing**: Data is generated in batches as PyArrow `RecordBatch` objects through the `process_images_detect` and `process_images_cls` functions, then written into LanceDB.

5. **Data Reading**: Data is read from LanceDB using the `LanceDataset` class, where the `load_data_list` method reads all data from the database, and the `get_data_info` method obtains detailed information for individual data items.

6. **Data Decoding**: In the `get_data_info` method, `bbox` and `catid` fields stored as byte data are decoded back into NumPy arrays for further processing.

Through this method, the dataset's storage structure in LanceDB is clear and efficient for reading and processing.
