# FOMO Model Training

This section will introduce how to train the FOMO face mask detection model on the COCO MASK dataset. The FOMO face mask detection model is implemented based on MobileNet V2 and MobileNet V3 (the actual neural network used depends on the model configuration file you choose). For more information about MobileNet, please refer to the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf).

## Dataset Preparation

Before training the FOMO model, we need to prepare the dataset. Here, we take the already annotated face mask COCO dataset as an example, which you can download from [SSCMA - Public Datasets](../../datasets/public#obtaining-public-datasets).

## Model Selection and Training

SSCMA offers various FOMO model configurations, and you can choose the appropriate model for training based on your needs.

```sh
fomo_mobnetv2_0.35_abl_coco.py
fomo_mobnetv2_1_x16_coco.py
```

Here, we take `fomo_mobnetv2_0.35_abl_coco.py` as an example to show how to use SSCMA for FOMO model training.

```sh
python3 tools/train.py \
    configs/fomo/fomo_mobnetv2_0.35_abl_coco.py \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/ \
    epochs=50 \
    height=192 \
    width=192
```

- `configs/fomo/fomo_mobnetv2_0.35_abl_coco.py`: Specifies the configuration file, defining the model and training settings.
- `--cfg-options`: Used to specify additional configuration options.
    - `data_root`: Sets the root directory of the dataset.
    - `num_classes`: Specifies the number of categories the model needs to recognize.
    - `train_ann`: Specifies the path to the annotation file for training data.
    - `val_ann`: Specifies the path to the annotation file for validation data.
    - `train_data`: Specifies the prefix path for training images.
    - `val_data`: Specifies the prefix path for validation images.
    - `epochs`: Sets the maximum number of training epochs.

After the training is complete, you can find the trained model in the `work_dirs/fomo_mobnetv2_0.35_abl_coco` directory. Before looking for the model, we suggest focusing on the training results first. Below is an analysis of the results and some suggestions for improvement.

:::details

```sh
12/16 04:32:12 - mmengine - INFO - Epoch(val) [100][6/6]    P: 0.0000  R: 0.0000  F1: 0.0000  data_time: 0.0664  time: 0.0796
```

The F1 score combines the precision and recall metrics, aiming to provide a single number to measure the overall performance of the model. The F1 score ranges from 0 to 1, with higher values indicating higher precision and recall, and better performance. The F1 score reaches its maximum value when the precision and recall of the model are equal.

:::

## Model Exporting and Verification

During the training process, you can view the training logs, export the model, and verify the model's performance at any time. Some of the metrics output during model verification are also displayed during the training process. Therefore, in this part, we will first introduce how to export the model and then discuss how to verify the accuracy of the exported model.

### Exporting the Model

Here, we take exporting the TFLite model as an example. You can use the following command to export TFLite models of different accuracies:

```sh
python3 tools/export.py \
    configs/fomo/fomo_mobnetv2_0.35_abl_coco.py \
    work_dirs/epoch_50.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/ \
    epochs=50 \
    --imgsz 192 192 \
    --format tflite \
    --image_path $(pwd)/datasets/coco_mask/mask/valid
```

:::warning

We recommend using the same resolution for training and exporting. In the current situation, using different resolutions for training and exporting may result in reduced model accuracy or complete loss of accuracy.

:::

:::tip

During the export process, an internet connection may be required to install certain dependencies. If you cannot access the internet, please ensure that the following dependencies are already installed in the current Python environment:

```
tensorflow
hailo_sdk_client
onnx
onnx2tf
tf-keras
onnx-graphsurgeon
sng4onnx
onnxsim
```

In addition, `onnx2tf` may also need to download calibration-related data during runtime. You can refer to the following link to download it in advance to the SSCMA root directory.

```sh
wget https://github.com/PINTO0309/onnx2tf/releases/download/1.20.4/calibration_image_sample_data_20x128x128x3_float32.npy  \
    -O calibration_image_sample_data_20x128x128x3_float32.npy
```

:::

### Verifying the Model

After exporting the model, you can use the following command to verify its performance:

```sh
python3 tools/test.py \
    configs/fomo/fomo_mobnetv2_0.35_abl_coco.py \
    work_dirs/epoch_50_int8.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/
```

### QAT

QAT (Quantization-Aware Training) is a method that simulates quantization operations during the model training process, allowing the model to gradually adapt to quantization errors, thereby maintaining high accuracy after quantization. SSCMA supports QAT, and you can refer to the following method to obtain a QAT model and verify it again.

```sh
python3 tools/quantization.py \
    configs/fomo/fomo_mobnetv2_0.35_abl_coco.py \
    work_dirs/epoch_50.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/ \
    epochs=50 \
    height=192 \
    width=192
```

After QAT training, the quantized model will be automatically exported, and its storage path will be `out/qat_model_test.tflite`. You can use the following command to verify it:

```sh
python3 tools/test.py \
    configs/fomo/fomo_mobnetv2_0.35_abl_coco.py \
    out/qat_model_test.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann=train/_annotations.coco.json \
    val_ann=valid/_annotations.coco.json \
    train_data=train/ \
    val_data=valid/
```
