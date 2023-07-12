# YOLO Model Training

This section describes how to train the digital meter model on the COCO digital meter datasets. The implementations of yolo digital meter detection model is based on the [yolov5](https://github.com/ultralytics/yolov5) and power by [mmyolo](https://github.com/open-mmlab/mmyolo)

## Prepare Datasets

EdgeLab uses [Digital Meter Datasets](https://universe.roboflow.com/seeeddatasets/seeed_meter_digit/) by default to train the yolov5 model, please refer to the following steps to complete the preparation of datasets.

1. Download digital meter datasets with COCO datasets mode

2. Remember its **folder path** (e.g. `datasets\digital_meter`) of the unpacked datasets, you may need to use this folder path later.

## Choose a Configuration

We will choose a appropriate configuration file depending on the type of training task we need to perform, which we have already introduced in [Config](../config.md), for a brief description of the functions, structure, and principles of the configuration file.

For the yolov5 model example, we use `yolov5_tiny_1xb16_300e_coco.py` as the configuration file, which is located in the folder under the EdgeLab root directory `configs/yolov5` and its additionally inherits the `base_arch.py` configuration file.

For beginners, we recommend to pay attention to the `data_root` and `epochs` parameters in this configuration file at first.

::: details `yolov5_tiny_1xb16_300e_coco.py`

```python
_base_='../_base_/default_runtime_det.py'
_base_ = ["./base_arch.py"]

anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]
num_classes = 11
deepen_factor = 0.33
widen_factor = 0.15

strides = [8, 16, 32]

model = dict(
    type='mmyolo.YOLODetector',
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
    ),
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
        ),
    ),
)
```

:::

## Training Model

Training the model requires using our previously configured EdgeLab working environment, if you follow our [Installation](../../introduction/installation.md) guide using Conda to install EdgeLab in a virtual environment named `edgelab`, please first make sure that you are currently in the virtual environment.

Then, in the EdgeLab project root directory, we execute the following command to train a yolov5 digital meter detection model.

```sh
python3 tools/train.py \
    configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
    --cfg-options \
        data_root='datasets/digital_meter' \
        epochs=50
```

During training, the model weights and related log information are saved to the path `work_dirs/yolov5_tiny_1xb16_300e_coco` by default, and you can use tools such as [TensorBoard](https://www.tensorflow.org/tensorboard/get_started) to monitor for training.

```sh
tensorboard --logdir work_dirs/yolov5_tiny_1xb16_300e_coco
```

After the training is completed, the path of the latest yolov5 model weights file is saved in the `work_dirs/yolov5_tiny_1xb16_300e_coco/last_checkpoint` file. Please take care of the path of the weight file, as it is needed when converting the model to other formats.

::: tip

If you have a virtual environment configured but not activated, you can activate it with the following command.

```sh
conda activate edgelab
```

:::

## Testing and Evaluation

### Testing

After have finished training the yolov5 model, you can specify specific weights and test the model using the following command.

```sh
python3 tools/inference.py \
    configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
    "$(cat work_dirs/yolov5_tiny_1xb16_300e_coco/last_checkpoint)" \
    --show \
    --cfg-options \
        data_root='datasets/digital_meter'
```

::: tip

If you want a real-time preview while testing, you can append a parameter `--show` to the test command to show the predicted results. For more optional parameters, please refer to the source code `tools/inference.py`.

:::

### Evaluation

In order to further test and evaluate the model on a realistic edge computing device, you need to export the model. In the process of exporting the model, EdgeLab will do some optimization on the model, such as model pruning, distillation, etc. You can refer to the [Export](../export/overview) section to learn more about how to export models.

### Deployment

After exporting the model, you can deploy the model to the edge computing device for testing and evaluation. You can refer to the [Deploy](../../deploy/examples.md) section to learn more about how to deploy models.
