# Training and Validation Pipeline

In the training and validation processes, we need to perform a series of operations such as data preprocessing, feature engineering, model training, and model evaluation. To facilitate these operations for users, we provide a complete training and validation pipeline, which can be completed with simple configurations.

:::tip

In the training process, the dataset and data loader are core components. SSCMA is based on MMEngine, so the concepts are the same as in PyTorch. The dataset is responsible for defining the data volume, reading, and preprocessing methods, while the data loader is responsible for iteratively loading data in batches, with settings such as batch size, random shuffling, and parallelism. For more information, please refer to [PyTorch Data Loading](https://pytorch.org/docs/stable/data.html). If you are unable to access this link due to network issues, it may be related to the validity of the web page link or network connectivity. Please check the legitimacy of the web page links and try again if necessary.

:::

In the executor (Runner) of MMEngine, you can specify data loaders for different training stages by setting three specific parameters:

- `train_dataloader`: This parameter is called in the Runner.train() method of the executor, responsible for providing data required for model training.
- `val_dataloader`: This parameter is used in the Runner.val() method of the executor and is called at specific training epochs within the Runner.train() method for model performance validation and evaluation.
- `test_dataloader`: This parameter is used in the Runner.test() method of the executor for the final testing of the model.

MMEngine is compatible with PyTorch's native DataLoader, which means you can directly pass the already created DataLoader instances to the above three parameters.

Training pipeline configuration typically includes the following steps:

1. **Load Image**: Define how to read images from files, including the decoding backend and related parameters.
2. **Load Annotations**: Define how to read annotation information corresponding to images, including whether to include bounding boxes.
3. **Data Augmentation**: Define various data augmentation operations, such as random HSV augmentation, Mosaic augmentation, random size adjustment, random cropping, random flipping, padding, and MixUp augmentation.
4. **Data Packing**: Pack the processed images and annotation information into the model input format.

Taking RTMDet as an example, we will show how to configure the data loader in the following example.

### Load Image

The load image module is used to read images from files.

```python
dict(
    type=LoadImageFromFile,  # Type of image loading
    imdecode_backend="pillow",  # Decoding backend
    backend_args=None,  # Backend arguments
)
```

### Load Annotations

The load annotations module is used to read annotation information corresponding to images.

```python
dict(
    type=LoadAnnotations,  # Type of annotation loading
    imdecode_backend="pillow",  # Decoding backend
    with_bbox=True,  # Whether to include bounding boxes
)
```

### Data Augmentation

The data augmentation module is used to perform various augmentation operations on images to improve the model's generalization ability.

```python
dict(type=HSVRandomAug),  # Random HSV augmentation

dict(
    type=Mosaic,  # Mosaic augmentation
    img_scale=imgsz,  # Image size
    pad_val=114.0,  # Padding value
),

dict(
    type=RandomResize,  # Random size adjustment
    scale=(imgsz[0] * 2, imgsz[1] * 2),  # Size range
    ratio_range=(0.1, 2.0),  # Ratio range
    resize_type=Resize,  # Resize type
    keep_ratio=True,  # Whether to keep the ratio
),

dict(
    type=RandomCrop,  # Random cropping
    crop_size=imgsz,  # Crop size
),

dict(
    type=RandomFlip,  # Random flipping
    prob=0.5,  # Flipping probability
),

dict(
    type=Pad,  # Padding
    size=imgsz,  # Padding size
    pad_val=dict(img=(114, 114, 114)),  # Padding value
),

dict(
    type=MixUp,  # MixUp augmentation
    img_scale=imgsz,  # Image size
    ratio_range=(1.0, 1.0),  # Ratio range
    max_cached_images=20,  # Maximum number of cached images
    pad_val=114.0,  # Padding value
),
```

### Data Packing

The data packing module is used to pack the processed images and annotation information into the model input format.

```python
dict(type=PackDetInputs),  # Data packing
```
