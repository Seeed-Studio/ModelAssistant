# Model Configuration

EdgeLab uses the configuration processing system provided by [OpenMMLab - MMEngine](https://github.com/open-mmlab/mmengine) with a modular and inheritable design that provides users a unified configuration access interface for various tests and validations of different neural networks.


## Directory Structure

The configuration files used by EdgeLab are located in the `configs` directory, which are used for training different models under different tasks. And we have created many subfolders according to different tasks, and in each subfolder, different training pipeline parameters of multiple models are stored.

::: code-group

```sh [Directory Stracture]
configs
├── _base_
│   ├── datasets
│   │   └── coco_detection.py
│   ├── default_runtime_cls.py
│   ├── default_runtime_det.py
│   ├── default_runtime_pose.py
│   └── schedules
│       ├── schedule_1x.py
│       ├── schedule_20e.py
│       └── schedule_2x.py
├── fomo
│   ├── fomo_efficientnet_b0_x8_abl_coco.py
│   ├── fomo_mobnetv2_0.35_x8_abl_coco.py
│   ├── fomo_mobnetv2_fpn_0.35_x8_abl.py
│   ├── fomo_mobnetv2_x8_coco.py
│   ├── fomo_mobnetv2_x8_voc.py
│   ├── fomo_mobnetv3_0.35_x8_abl_coco.py
│   ├── fomo_shufflenetv2_0.1_x8_abl_coco.py
│   └── fomo_squeezenet_0.1_x8_abl_coco.py
├── pfld
│   └── pfld_mv2n_112.py
└── <Other Tasks...>
```

```sh [Subfolders for Different Tasks]
configs // [!code focus]
├── _base_ // [!code focus]
│   ├── datasets
│   │   └── coco_detection.py
│   ├── default_runtime_cls.py
│   ├── default_runtime_det.py
│   ├── default_runtime_pose.py
│   └── schedules
│       ├── schedule_1x.py
│       ├── schedule_20e.py
│       └── schedule_2x.py
├── fomo // [!code focus]
│   ├── fomo_efficientnet_b0_x8_abl_coco.py
│   ├── fomo_mobnetv2_0.35_x8_abl_coco.py
│   ├── fomo_mobnetv2_fpn_0.35_x8_abl.py
│   ├── fomo_mobnetv2_x8_coco.py
│   ├── fomo_mobnetv2_x8_voc.py
│   ├── fomo_mobnetv3_0.35_x8_abl_coco.py
│   ├── fomo_shufflenetv2_0.1_x8_abl_coco.py
│   └── fomo_squeezenet_0.1_x8_abl_coco.py
├── pfld // [!code focus]
│   └── pfld_mv2n_112.py
└── <Other Tasks...> // [!code focus]
```

```sh [Different Train Pipeline for each Tasks (e.g. FOMO)]
configs // [!code focus]
├── _base_
│   ├── datasets
│   │   └── coco_detection.py
│   ├── default_runtime_cls.py
│   ├── default_runtime_det.py
│   ├── default_runtime_pose.py
│   └── schedules
│       ├── schedule_1x.py
│       ├── schedule_20e.py
│       └── schedule_2x.py
├── fomo // [!code focus]
│   ├── fomo_efficientnet_b0_x8_abl_coco.py // [!code focus]
│   ├── fomo_mobnetv2_0.35_x8_abl_coco.py // [!code focus]
│   ├── fomo_mobnetv2_fpn_0.35_x8_abl.py // [!code focus]
│   ├── fomo_mobnetv2_x8_coco.py // [!code focus]
│   ├── fomo_mobnetv2_x8_voc.py // [!code focus]
│   ├── fomo_mobnetv3_0.35_x8_abl_coco.py // [!code focus]
│   ├── fomo_shufflenetv2_0.1_x8_abl_coco.py // [!code focus]
│   └── fomo_squeezenet_0.1_x8_abl_coco.py // [!code focus]
├── pfld
│   └── pfld_mv2n_112.py
└── <Other Tasks...>
```

:::

::: tip

The task folder named `_base_` is an inheritance object for other tasks. For more details about configuration file inheritance, please refer to [MMEngine - Configuration File Inheritance](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#id3).

:::


## Configuration Structure

Take the `fomo_mobnetv2_0.35_x8_abl_coco.py` configuration file as an example, we introduce different fields in this configuration file according to the different functional modules.

### Important Parameters

When changing the training configuration, it is usually necessary to modify the following parameters. For example, the `height` and `width` factors are usually for image size. So we recommend defining these parameters separately in the configuration file.

```python
height=96       # Input image height
width=96        # Input image width
batch_size=16   # Batch size of a single GPU during validation
workers=4       # Worker to pre-fetch data for each single GPU during validation
epoches=300     # Maximum training epochs: 300 epochs
lr=0.001        # Learn rate
```

### Model Config

In the configuration file of the FOMO model, we use model to set up detection algorithm components, include neural network components such as backbone, neck, etc. Part of model configuration as below:

```python
num_classes=2                                   # Number of class
model=dict(
    type='Fomo',                                # The name of detector
    backbone=dict(
        type='MobileNetV2',
        widen_factor=0.35,
        out_indices=(2, )),                     # The config of backbone
    head=dict(
        type='Fomo_Head',                       # The config of head
        input_channels=16,                      # The input channels, this is consistent with the input channels of neck
        num_classes=num_classes,                # Number of classes for classification
        middle_channels=[96, 32],               # The output channels for head conv
        act_cfg='ReLU6',                        # The config of activation function
        loss_cls=dict(type='BCEWithLogitsLoss', # This loss combines a Sigmoid layer and the BCELoss in one single class
                      reduction='none',
                      pos_weight=40),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
        cls_weight=40)                          # Parameter for pos_weight
)
```

### Dataset and Evaluator Config

Dataset and data pipeline need to be set to build the dataloader. Due to the complexity of this part, we use intermediate variables to simplify the writing of dataloader configs. More complex data argumentation methods can be found in `edgelab/datasets/pipelines` path.

We will demonstrate here the training and testing pipeline for FOMO, which uses the [Custom COCO_MASK Dataset](./datasets):


```python
dataset_type='FomoDatasets'   # Dataset type, this will be used to define the dataset
data_root=''                  # Root path of data
train_pipeline=[              # Training data loading pipeline
    dict(type='RandomResizedCrop', height=height, width=width, scale=(0.90, 1.1),
         p=1),                                 # RandomResizedCrop augmentation in albumentation for fomo
    dict(type='Rotate', limit=20),             # Rotate transform with limit degree 20
    dict(type='RandomBrightnessContrast',      # RandomBrightnessContrast augmentation in albumentation
         brightness_limit=0.2,                 # Factor range for changing brightness
         contrast_limit=0.2,                   # Factor range for changing contrast
         p=0.5),                               # Probability of applying the transform
    dict(type='HorizontalFlip', p=0.5),        # Flip the input horizontally around the y-axis
]
test_pipeline=[dict(type='Resize', height=height, width=width,
                    p=1)]                      # Resize the input to the given height and width

data=dict(samples_per_gpu=batch_size,          # Batch size of a single GPU during training
          workers_per_gpu=workers,             # Worker to pre-fetch data for each single GPU during training
          train_dataloader=dict(collate=True), # Flag of merging a list of samples to form a mini-batch
          val_dataloader=dict(collate=True),
          train=dict(type=dataset_type,
                     data_root=data_root,
                     ann_file='annotations/person_keypoints_train2017.json',
                     img_prefix='train2017',   # Path of annotation file and prefix of image path
                     pipeline=train_pipeline),
          val=dict(type=dataset_type,
                   data_root=data_root,
                   test_mode=True,             # Enable test mode of the dataset to avoid filtering annotations or images
                   ann_file='annotations/person_keypoints_val2017.json',
                   img_prefix='val2017',
                   pipeline=test_pipeline),
          test=dict(type=dataset_type,
                    data_root=data_root,
                    test_mode=True,
                    ann_file='annotations/person_keypoints_val2017.json',
                    img_prefix='val2017',
                    pipeline=test_pipeline))
```

Evaluators are used to compute the metrics of the trained model on the validation and testing datasets. The config of evaluators consists of one or a list of metric configs:

```python
evaluation=dict(interval=1, metric=['mAP'], fomo=True) # Validation metric for evaluate mAP
find_unused_parameters=True
```

### Optimizer Config

```python
optimizer=dict(type='Adam', lr=lr, weight_decay=0.0005)         # Adam gradient descent optimizer with base learning rate and weight decay
optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2)) # Config used to build the optimizer hook
```

:::tip

For more details on the application of Hook, please refer to [MMEngine - Hook](https://mmengine.readthedocs.io/en/latest/tutorials/hook.html).

:::

### Config File Inheritance

The directory `config/_base_` contains the default configuration file, and the configuration file are composed of the components in `_base_`, which is called the primitive.  

For easy testing, we recommend that users inherit the existing configuration files. For example, the training configuration file of a FOMO model with `_base_='. /_base_/default_runtime_det.py'`, and then based on the inherited file, we modify the necessary fields in the configuration file.

```python
_base_='../_base_/default_runtime_det.py
checkpoint_config=dict(interval=5) # Config to set the checkpoint hook
log_config=dict(                   # Config to register logger hook
            interval=150,          # Interval to print the log
            hooks=[
                dict(type='TextLoggerHook', ndigits=4),       # TXT logger
                dict(type='TensorboardLoggerHook', ndigits=4) # Tensorboard logger
            ])                                                # The logger used to record the training process
epochs=300
runner=dict(type='EpochBasedRunner',  # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
            max_epochs=epochs)        # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
dist_params=dict(backend='nccl')      # Parameters to setup distributed training, the port can also be set
log_level = 'INFO'                    # The level of logging
load_from = None                      # Load models as a pre-trained model from a given path, this will not resume training
resume_from = None                    # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved
workflow = [('train', 1)]             # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 300 epochs according to the total_epochs
opencv_num_threads = 1                # Disable OpenCV multi-threads to save memory
work_dir = './work_dirs'              # Directory to save the model checkpoints and logs for the current experiments
```


## Parameterized Configuration

When submitting a job using `tools/train.py` or `tools/test.py` from EdgeLab, you can specify `--cfg-options` to temporarily overwrite the configuration.

::: tip

You can specify configuration options in the order of the dict keys in the original configuration and update the dict chain of configuration keys. For example, `--cfg-options data_root='. /dataset/coco'` change the data root directory of the dataset.

:::


## FAQs

- The configuration file of different models will be different, how do I understand it?

    For more details, please refer to [MMDet Config](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html), [MMPose Config](https://mmpose.readthedocs.io/en/latest/tutorials/0_config.html) and [MMCls Config](https://mmclassification.readthedocs.io/en/latest/tutorials/config.html).
