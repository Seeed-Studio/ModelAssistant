# PFLD Model Training

This section describes how to train the PFLD model on the PFLD meter dataset. The PFLD model is presented in the paper [PFLD: A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf).

## Prepare Datasets

EdgeLab uses [Custom Meter Datasets](../datasets.md#EdgeLab) by default to train the PFLD model, please refer to the following steps to complete the preparation of datasets.

1. Please refer to [Internet Datasets - EdgeLab - Custom Meter Dataset](../datasets.md#EdgeLab) to download and unpack the dataset.

2. Remember its **folder path** (e.g. `datasets\meter`) of the unpacked datasets, you may need to use this folder path later.

## Choose a Configuration

We will choose a appropriate configuration file depending on the type of training task we need to perform, which we have already introduced in [Config](../config.md), for a brief description of the functions, structure, and principles of the configuration file.

For the meter PFLD model example, we use `pfld_mbv2n_112.py` as the configuration file, which is located in the folder under the EdgeLab root directory `configs/pfld` and its additionally inherits the `default_runtime_pose.py` configuration file.

For beginners, we recommend to pay attention to the `data_root` and `epochs` parameters in this configuration file at first.

::: details `pfld_mbv2n_112.py`

```python
_base_='../_base_/default_runtime_pose.py'

num_classes=1
model=dict(type='PFLD',
             backbone=dict(type='PfldMobileNetV2',
                           inchannel=3,
                           layer1=[16, 16, 16, 16, 16],
                           layer2=[32, 32, 32, 32, 32, 32],
                           out_channel=16),
             head=dict(type='PFLDhead',
                       num_point=num_classes,
                       input_channel=16,
                       loss_cfg=dict(type='L1Loss')))

# dataset settings
dataset_type='MeterData'

data_root=''
height=112
width=112
batch_size=32
workers=4

train_pipeline=[
    dict(type="Resize", height=height, width=width, interpolation=0),
    dict(type='ColorJitter', brightness=0.3, p=0.5),
    dict(type='GaussNoise'),
    dict(type='MedianBlur', blur_limit=3, p=0.3),
    dict(type='HorizontalFlip'),
    dict(type='VerticalFlip'),
    dict(type='Rotate'),
    dict(type='Affine', translate_percent=[0.05, 0.1], p=0.6)
]

val_pipeline=[dict(type="Resize", height=height, width=width)]

train_dataloader=dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 index_file=r'train/annotations.txt',
                 pipeline=train_pipeline,
                 test_mode=False),
)

val_dataloader=dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    collate_fn=dict(type='default_collate'),
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(type=dataset_type,
                 data_root=data_root,
                 index_file=r'val/annotations.txt',
                 pipeline=val_pipeline,
                 test_mode=True),
)
test_dataloader=val_dataloader

lr=0.0001
epochs=300
evaluation=dict(save_best='loss')
optim_wrapper=dict(
    optimizer=dict(type='Adam', lr=lr, betas=(0.9, 0.99), weight_decay=1e-6))
optimizer_config=dict(grad_clip=dict(max_norm=35, norm_type=2))
val_evaluator=dict(type='PointMetric')
test_evaluator=val_evaluator
find_unused_parameters=True
train_cfg=dict(by_epoch=True, max_epochs=500)

# learning policy
param_scheduler=[
    dict(type='LinearLR', begin=0, end=500, start_factor=0.001,
         by_epoch=False),  # warm-up
    dict(type='MultiStepLR',
         begin=1,
         end=500,
         milestones=[350, 400, 450, 490],
         gamma=0.1,
         by_epoch=True)
]
```

:::

## Training Model

Training the model requires using our previously configured EdgeLab working environment, if you follow our [Installation](../../introduction/installation.md) guide using Conda to install EdgeLab in a virtual environment named `edgelab`, please first make sure that you are currently in the virtual environment.

Then, in the EdgeLab project root directory, we execute the following command to train an end-to-end meter PFLD model.

```sh
python3 tools/train.py \
    configs/pfld/pfld_mbv2n_112.py \
    --cfg-options \
        data_root='datasets/meter' \
        epochs=50
```

During training, the model weights and related log information are saved to the path `work_dirs/pfld_mbv2n_112` by default, and you can use tools such as [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) to monitor for training.

```sh
tensorboard --logdir work_dirs/pfld_mbv2n_112
```

After the training is completed, the path of the latest FOMO model weights file is saved in the `work_dirs/pfld_mbv2n_112/last_checkpoint` file. Please take care of the path of the weight file, as it is needed when converting the model to other formats.

::: tip

If you have a virtual environment configured but not activated, you can activate it with the following command.

```sh
conda activate edgelab
```

:::

## Testing and Evaluation

### Testing

After have finished training the PFLD model, you can specify specific weights and test the model using the following command.

```sh
python3 tools/inference.py \
    configs/pfld/pfld_mbv2n_112.py \
    "$(cat work_dirs/pfld_mbv2n_112/last_checkpoint)" \
    --cfg-options \
        data_root='datasets/meter'
```

::: tip

If you want a real-time preview while testing, you can append a parameter `--show` to the test command to show the predicted results. For more optional parameters, please refer to the source code `tools/test.py`.

:::

### Evaluation

In order to further test and evaluate the model on a realistic edge computing device, you need to export the model. In the process of exporting the model, EdgeLab will do some optimization on the model, such as model pruning, distillation, etc. You can refer to the [Export](../export/overview) section to learn more about how to export models.

### Deployment

After exporting the model, you can deploy the model to the edge computing device for testing and evaluation. You can refer to the [examples](../../examples/examples.md) section to learn more about how to deploy models.
