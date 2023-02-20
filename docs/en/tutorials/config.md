#  Config

- [Config](#config)
    - [Directory structure for configuration files](#directory-structure-for-configuration-files)
    - [Config file content](#config-file-content)
        - [Important parameters](#important-parameters)
        - [Model config](#model-config)
        - [Dataset and evaluator config](#dataset-and-evaluator-config)
        - [Optimization config](#optimization-config)
        - [Config file inheritance](#config-file-inheritance)
    - [Modify config through script arguments](#modify-config-through-script-arguments)
    - [FAQs](#faqs)

Edgelab repository use config system offered by OpenMMLab. It has a modular and inheritance design, which is convenient to conduct various experiments. You can inspect the config file [here](../../../configs/).


## Directory structure for configuration files

The directory structure for config files in our repository follows as below:
```sh
configs  
├── audio_classify  
│   ├── ali_classiyf_small_8k_8192.py  
│   └── README.md  
├── _base_  
│   ├── datasets  
│   │   └── coco_detection.py  
│   ├── default_runtime.py  
│   └── schedules  
│       ├── schedule_1x.py  
│       ├── schedule_20e.py  
│       └── schedule_2x.py  
├── fastestdet  
│   ├── fastestdet_shuffv2_spp_fomo_voc.py  
│   └── fastestdet_shuffv2_spp_voc.py  
├── fomo  
│   ├── fomo_mobnetv2_0.35_x8_abl_coco.py  
│   ├── fomo_mobnetv2_x8_coco.py  
│   └── fomo_mobnetv2_x8_voc.py  
├── pfld  
│   ├── pfld_mv2n_112.py  
│   └── README.md  
└── yolo  
    ├── README.md  
    └── yolov3_mbv2_416_voc.py  
```

## Config file content

Taking [fomo_mobnetv2_0.35_x8_abl_coco.py](../../../configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py) as an example, we will introduce each field in the config according to different function modules:

### Important parameters

When changing the training configuration, it is usually necessary to modify the following parameters. For example, the `height` and `width` factors are usually for image size. So we recommend defining these parameters separately in the configuration file.

```sh
height=96       # Input image height
width=96        # Input image width
batch_size=16   # Batch size of a single GPU during validation
workers=4       # Worker to pre-fetch data for each single GPU during validation
epoches=300     # Maximum training epochs: 300 epochs
lr=0.001        # Learn rate
```

### Model config

In our repository, we use model to set up detection algorithm components, include neural network components such as backbone, neck, etc. Part of model configuration as below:
```sh
num_classes=2   # Number of class
model = dict(
    type='Fomo',    # The name of detector
    backbone=dict(type='MobileNetV2', widen_factor=0.35, out_indices=(2, )),    # The config of backbone
    head=dict(
        type='Fomo_Head',   # The config of head
        input_channels=16,  # The input channels, this is consistent with the input channels of neck
        num_classes=num_classes,    # Number of classes for classification
        middle_channels=[96, 32],   # The output channels for head conv
        act_cfg='ReLU6',    # The config of activation function
        loss_cls=dict(type='BCEWithLogitsLoss', # This loss combines a Sigmoid layer and the BCELoss in one single class
                      reduction='none',
                      pos_weight=40),
        loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
        cls_weight=40,  # Parameter for pos_weight
    ),
)
```

### Dataset and evaluator config

Dataset and data pipeline need to be set to build the dataloader. Due to the complexity of this part, we use intermediate variables to simplify the writing of dataloader configs. More complex data argumentation methods can be found in [pipiline](../../../edgelab/datasets/pipelines/). We will demonstrate the training and testing dataflow for fomo here.
```sh
dataset_type = 'FomoDatasets'   # Dataset type, this will be used to define the dataset
data_root = ''  # Root path of data
train_pipeline = [  # Training data loading pipeline
    dict(type='RandomResizedCrop', height=height, width=width, scale=(0.90, 1.1),
         p=1),  # RandomResizedCrop augmentation in albumentation for fomo
    dict(type='Rotate', limit=20), # Rotate transform with limit degree 20
    dict(type='RandomBrightnessContrast',   # RandomBrightnessContrast augmentation in albumentation
         brightness_limit=0.2,  # factor range for changing brightness
         contrast_limit=0.2,    # factor range for changing contrast
         p=0.5),    # probability of applying the transform
    dict(type='HorizontalFlip', p=0.5), # Flip the input horizontally around the y-axis
]
test_pipeline = [dict(type='Resize', height=height, width=width, p=1)] # Resize the input to the given height and width

data = dict(    # Train dataloader config
            samples_per_gpu=batch_size, # Batch size of a single GPU during training
            workers_per_gpu=workers,    # Worker to pre-fetch data for each single GPU during training
            train_dataloader=dict(collate=True),   # Flag of merging a list of samples to form a mini-batch
            val_dataloader=dict(collate=True),
            train=dict(type=dataset_type,
                       data_root=data_root,
                       ann_file='annotations/person_keypoints_train2017.json',  # Path of annotation file
                       img_prefix='train2017',  # Prefix of image path
                       pipeline=train_pipeline),
            val=dict(type=dataset_type,
                     data_root=data_root,
                     test_mode=True, # Turn on test mode of the dataset to avoid filtering annotations or images
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
```sh
evaluation = dict(interval=1, metric=['mAP'], fomo=True)    # Validation evaluator config, metric for evaluate mAP
find_unused_parameters = True
```

### Optimization config

```sh
optimizer = dict(type='Adam', lr=lr, weight_decay=0.0005) # Adam gradient descent optimizer with base learning rate and weight decay

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))   # Config used to build the optimizer hook, refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/optimizer.py#L8 for implementation details
```

### Config file inheritance

`config/_base_/default_runtime.py` contains default runtime. The configs that are composed of components from _base_ are called primitive.  
For easy understanding, we recommend user inherit from existing methods. For example, set `__base__='../_base_/default_runtime.py'` in fomo config file, then modify the necessary fields in the config files.
```sh
checkpoint_config = dict(interval=5)    # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation. The save interval is 1
log_config = dict(  # config to register logger hook
    interval=150,   # Interval to print the log
    hooks=[
        # dict(type='TextLoggerHook', ndigits=4),   # test logger
        dict(type='TensorboardLoggerHook', ndigits=4)   # tensorboard logger
    ])  # The logger used to record the training process

epochs=300
runner = dict(type='EpochBasedRunner',  # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
              max_epochs=epochs)    # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`

dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set

log_level = 'INFO'  # The level of logging

load_from = None    # load models as a pre-trained model from a given path. This will not resume training
resume_from = None  # Resume checkpoints from a given path, the training will be resumed from the epoch when the checkpoint's is saved

workflow = [('train', 1)]   # Workflow for runner. [('train', 1)] means there is only one workflow and the workflow named 'train' is executed once. The workflow trains the model by 300 epochs according to the total_epochs

opencv_num_threads = 1  # # Enable opencv multi-threads

# set multi-process start method as `fork` to speed up the training
# mp_start_method = 'fork'

work_dir = './work_dirs'    # Directory to save the model checkpoints and logs for the current experiments
```

## Modify config through script arguments

When submitting jobs using tools/train.py or tools/test.py, you may specify --cfg-options to in-place modify the config.  

- Update config keys of dict chains.  
The config options can be specified following the order of the dict keys in the original config. For    example, `--cfg-options data_root='./dataset/coco'` changes data root for dataset.


## FAQs
Configuration files are slightly different for different models, Please refer to [mmdet config](https://mmdetection.readthedocs.io/en/latest/tutorials/config.html), [mmpose config](https://mmpose.readthedocs.io/en/latest/tutorials/0_config.html) and [mmcls config](https://mmclassification.readthedocs.io/en/latest/tutorials/config.html) to see more details.