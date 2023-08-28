# FOMO Model Training

This section describes how to train the FOMO mask detection model on the COCO MASK datasets. the implementations of FOMO mask detection model is based on the MobileNet V2 and MobileNet V3 (the actual neural network selected depends on the model profile you choose).

For more information about MobileNet, please refer to the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf).

## Prepare Datasets

[SSCMA](https://github.com/Seeed-Studio/SSCMA) uses [COCO_MASK Datasets](../datasets.md#SSCMA) by default to train the FOMO model, please refer to the following steps to complete the preparation of datasets.

1. Please refer to [Internet Datasets - [SSCMA](https://github.com/Seeed-Studio/SSCMA) - COCO_MASK Dataset](../datasets.md#SSCMA) to download and unpack the dataset.

2. Remember its **folder path** (e.g. `datasets\mask`) of the unpacked datasets, you may need to use this folder path later.

## Choose a Configuration

We will choose a appropriate configuration file depending on the type of training task we need to perform, which we have already introduced in [Config](../config.md), for a brief description of the functions, structure, and principles of the configuration file.

For the FOMO model example, we use `fomo_mobnetv2_0.35_x8_abl_coco.py` as the configuration file, which is located in the folder under the [SSCMA](https://github.com/Seeed-Studio/SSCMA) root directory `configs/fomo` and its additionally inherits the `default_runtime_det.py` configuration file.

For beginners, we recommend to pay attention to the `data_root` and `epochs` parameters in this configuration file at first.

::: details `fomo_mobnetv2_0.35_x8_abl_coco.py`

```python
_base_='../_base_/default_runtime_det.py'
default_scope='sscma'
custom_imports=dict(imports=['sscma'], allow_failed_imports=False)

num_classes=2
model=dict(type='Fomo',
           backbone=dict(type='mmdet.MobileNetV2', widen_factor=0.35, out_indices=(2,)),
           head=dict(type='FomoHead',
                     input_channels=[16],
                     num_classes=num_classes,
                     middle_channel=48,
                     act_cfg='ReLU6',
                     loss_cls=dict(type='BCEWithLogitsLoss',
                                   reduction='none',
                                   pos_weight=40),
                     loss_bg=dict(type='BCEWithLogitsLoss', reduction='none'),
           ),
)

# dataset settings
dataset_type='FomoDatasets'
data_root=''
height=96
width=96
batch_size=16
workers=1

train_pipeline=[
    dict(type='RandomResizedCrop',
         height=height,
         width=width,
         scale=(0.80, 1.2),
         p=1),
    dict(type='Rotate', limit=30),
    dict(type='RandomBrightnessContrast',
         brightness_limit=0.3,
         contrast_limit=0.3,
         p=0.5),
    dict(type='HorizontalFlip', p=0.5),
]
test_pipeline=[dict(type='Resize', height=height, width=width, p=1)]

train_dataloader=dict(
    batch_size=batch_size,
    num_workers=workers,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='fomo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 ann_file='train/_annotations.coco.json',
                 img_prefix='train',
                 pipeline=train_pipeline),
)
val_dataloader=dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='fomo_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 ann_file='valid/_annotations.coco.json',
                 img_prefix='valid',
                 pipeline=test_pipeline))
test_dataloader=val_dataloader

# optimizer
lr=0.001
epochs=300
find_unused_parameters=True
optim_wrapper=dict(optimizer=dict(type='Adam', lr=lr, weight_decay=5e-4,eps=1e-7))

#evaluator
val_evaluator=dict(type='FomoMetric')
test_evaluator=val_evaluator
train_cfg=dict(by_epoch=True, max_epochs=70)

# learning policy
param_scheduler=[
    dict(type='LinearLR', begin=0, end=30, start_factor=0.001, by_epoch=False),  # warm-up
    dict(type='MultiStepLR',
         begin=1,
         end=500,
         milestones=[100, 200, 250],
         gamma=0.1,
         by_epoch=True)
]
```

:::

## Training Model

Training the model requires using our previously configured [SSCMA](https://github.com/Seeed-Studio/SSCMA) working environment, if you follow our [Installation](../../introduction/installation.md) guide using Conda to install [SSCMA](https://github.com/Seeed-Studio/SSCMA) in a virtual environment named `sscma`, please first make sure that you are currently in the virtual environment.

Then, in the [SSCMA](https://github.com/Seeed-Studio/SSCMA) project root directory, we execute the following command to train a FOMO mask detection model.

```sh
python3 tools/train.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    --cfg-options \
        data_root='datasets/mask' \
        epochs=50
```

During training, the model weights and related log information are saved to the path `work_dirs/fomo_mobnetv2_0.35_x8_abl_coco` by default, and you can use tools such as [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) to monitor for training.

```sh
tensorboard --logdir work_dirs/fomo_mobnetv2_0.35_x8_abl_coco
```

After the training is completed, the path of the latest FOMO model weights file is saved in the `work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint` file. Please take care of the path of the weight file, as it is needed when converting the model to other formats.

::: tip

If you have a virtual environment configured but not activated, you can activate it with the following command.

```sh
conda activate sscma
```

:::

## Testing and Evaluation

### Testing

After have finished training the FOMO model, you can specify specific weights and test the model using the following command.

```sh
python3 tools/inference.py \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    "$(cat work_dirs/fomo_mobnetv2_0.35_x8_abl_coco/last_checkpoint)" \
    --cfg-options \
        data_root='datasets/mask'
```

::: tip

If you want a real-time preview while testing, you can append a parameter `--show` to the test command to show the predicted results. For more optional parameters, please refer to the source code `tools/test.py`.

:::

### Evaluation

In order to further test and evaluate the model on a realistic edge computing device, you need to export the model. In the process of exporting the model, [SSCMA](https://github.com/Seeed-Studio/SSCMA) will do some optimization on the model, such as model pruning, distillation, etc. You can refer to the [Export](../export/overview) section to learn more about how to export models.

### Deployment

After exporting the model, you can deploy the model to the edge computing device for testing and evaluation. You can refer to the [Deploy](../../deploy/examples.md) section to learn more about how to deploy models.
