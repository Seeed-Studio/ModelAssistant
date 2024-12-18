# Model Configuration

When training deep learning tasks, we typically need to define a model to implement the main body of the algorithm. We use the `model` field to configure the model's related information, including the model's network structure, size, and the way each part connects. This is a modular design approach, allowing users to define models according to their needs, while the implementation of each module is defined in the core codebase of `sscma`.

:::tip

For the model core, SSCMA is developed based on MMEngine, and the definition of the model also follows the structure of MMEngine. It is managed by the executor and needs to implement train_step, val_step, and test_step methods. For deep learning tasks such as detection, recognition, and segmentation, the above methods are usually standard processes, such as updating parameters and returning losses in train_step; val_step and test_step return prediction results.
:::

Taking RTMDet as an example, we will explain the model configuration in parts.

## Data Preprocessing

Define the mean, standard deviation, color space conversion, and data augmentation strategies for input data. We can define the parameters for data preprocessing in the configuration file, such as mean, standard deviation, color space conversion, etc.

```python
data_preprocessor=dict(
    type=DetDataPreprocessor,  # Type of data preprocessor
    mean=[103.53, 116.28, 123.675],  # Mean values
    std=[57.375, 57.12, 58.395],  # Standard deviations
    bgr_to_rgb=False,  # Whether to convert BGR to RGB
    batch_augments=[  # Batch data augmentation
        dict(
            type=YOLOXBatchSyncRandomResize,  # Type of data augmentation
            random_size_range=(224, 1024),  # Random size range
            size_divisor=32,  # Size divisor
            interval=1,  # Interval
        )
    ],
)
```

## Backbone Network

The backbone network module is used to define the main structure of the model.

```python
backbone=dict(
    type=TimmBackbone,  # Type of backbone network
    model_name="mobilenetv4_conv_small.e2400_r224_in1k",  # Model name
    features_only=True,  # Whether to extract features only
    pretrained=True,  # Whether to use a pre-trained model
    out_indices=[2, 3, 4],  # Output feature layer indices
    init_cfg=None,  # Initialization configuration
)
```

## Neck Network

The neck network module is used to connect the backbone network and the head network.

```python
neck=dict(
    type=CSPNeXtPAFPN,  # Type of neck network
    deepen_factor=d_factor,  # Deepen factor
    widen_factor=1,  # Widen factor
    in_channels=[64, 96, 960],  # Input channel numbers
    out_channels=256,  # Output channel number
    num_csp_blocks=3,  # Number of CSP blocks
    expand_ratio=0.5,  # Expansion ratio
    norm_cfg=dict(type=SyncBatchNorm),  # Normalization configuration
    act_cfg=dict(type=SiLU, inplace=True),  # Activation function configuration
)
```

## Head Network

The bounding box head module is used to define the structure of the detection head and the loss function.

```python
bbox_head=dict(
    type=RTMDetHead,  # Type of bounding box head
    head_module=dict(
        type=RTMDetSepBNHeadModule,  # Type of head module
        num_classes=80,  # Number of classes
        in_channels=256,  # Input channel number
        stacked_convs=2,  # Number of stacked convolutional layers
        feat_channels=256,  # Feature channel number
        norm_cfg=dict(type=SyncBatchNorm),  # Normalization configuration
        act_cfg=dict(type=SiLU, inplace=True),  # Activation function configuration
        share_conv=True,  # Whether to share convolutional layers
        pred_kernel_size=1,  # Prediction kernel size
        featmap_strides=[8, 16, 32],  # Feature map strides
    ),
    prior_generator=dict(type=MlvlPointGenerator, offset=0, strides=[8, 16, 32]),  # Prior box generator
    bbox_coder=dict(type=DistancePointBBoxCoder),  # Bounding box coder
    loss_cls=dict(
        type=QualityFocalLoss, use_sigmoid=True, beta=2.0, loss_weight=1.0  # Classification loss
    ),
    loss_bbox=dict(type=GIoULoss, loss_weight=2.0),  # Bounding box loss
)
```

## Training Configuration

The training configuration module is used to define parameters and strategies during the training process.

```python
train_cfg=dict(
    assigner=dict(
        type=BatchDynamicSoftLabelAssigner,  # Type of assigner
        num_classes=num_classes,  # Number of classes
        topk=13,  # Top-k value
        iou_calculator=dict(type=BboxOverlaps2D),  # IOU calculator
    ),
    allowed_border=-1,  # Allowed border
    pos_weight=-1,  # Positive sample weight
    debug=False,  # Whether to enable debug
)
```

## Testing Configuration

The testing configuration module is used to define parameters and strategies during the testing process.

```python
test_cfg=dict(
    multi_label=True,  # Whether to use multi-label
    nms_pre=30000,  # Maximum number of boxes before NMS
    min_bbox_size=0,  # Minimum bounding box size
    score_thr=0.001,  # Score threshold
    nms=dict(type=nms, iou_threshold=0.65),  # NMS configuration
    max_per_img=300,  # Maximum number of detections per image
)
```
