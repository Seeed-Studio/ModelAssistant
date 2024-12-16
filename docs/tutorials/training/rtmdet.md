# RTMDet Model Training

RTMDet (Real-time Models for Object Detection) is a high-precision, low-latency single-stage object detection algorithm. The overall structure of the RTMDet model is almost identical to YOLOX, consisting of CSPNeXt + CSPNeXtPAFPN + SepBNHead with shared convolutional weights but separate BN calculations. The core internal module is also CSPLayer, but the Basic Block within it has been improved to CSPNeXt Block.

## Dataset Preparation

Before training the RTMDet model, we need to prepare the dataset. Here, we take the already annotated mask COCO dataset as an example, which you can download from [SSCMA - Public Datasets](../../datasets/public#obtaining-public-datasets).

## Model Selection and Training

SSCMA offers various RTMDet model configurations, and you can choose the appropriate model for training based on your needs.

```sh
rtmdet_l_8xb32_300e_coco.py
rtmdet_m_8xb32_300e_coco.py
rtmdet_mnv4_8xb32_300e_coco.py
rtmdet_nano_8xb32_300e_coco.py
rtmdet_nano_8xb32_300e_coco_relu.py
rtmdet_nano_8xb32_300e_coco_relu_q.py
rtmdet_s_8xb32_300e_coco.py
```

Here, we take `rtmdet_nano_8xb32_300e_coco.py` as an example to show how to use SSCMA for RTMDet model training.

```sh
python3 tools/train.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    max_epochs=50 \
    imgsz='(192,192)'
```

- `configs/rtmdet_nano_8xb32_300e_coco.py`: Specifies the configuration file, defining the model and training settings.
- `--cfg-options`: Used to specify additional configuration options.
    - `data_root`: Sets the root directory of the dataset.
    - `num_classes`: Specifies the number of categories the model needs to recognize.
    - `train_ann_file`: Specifies the path to the annotation file for training data.
    - `val_ann_file`: Specifies the path to the annotation file for validation data.
    - `train_img_prefix`: Specifies the prefix path for training images.
    - `val_img_prefix`: Specifies the prefix path for validation images.
    - `max_epochs`: Sets the maximum number of training epochs.
    - `imgsz`: Specifies the image size used for model training.

After the training is complete, you can find the trained model in the `work_dirs/rtmdet_nano_8xb32_300e_coco` directory. Before looking for the model, we suggest focusing on the training results first. Below is an analysis of the results and some suggestions for improvement.

:::details

```sh
12/15 08:45:34 - mmengine - INFO - Saving checkpoint at 50 epochs
12/15 08:45:35 - mmengine - INFO - Evaluating bbox...
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.32s).
Accumulating evaluation results...
DONE (t=0.06s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.254
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.743
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.093
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.259
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.422
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.456
12/15 08:45:36 - mmengine - INFO - bbox_mAP_copypaste: 0.254 0.743 0.093 -1.000 0.000 0.259
12/15 08:45:36 - mmengine - INFO - Epoch(val) [50][6/6]    coco/bbox_mAP: 0.2540  coco/bbox_mAP_50: 0.7430  coco/bbox_mAP_75: 0.0930  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: 0.0000  coco/bbox_mAP_l: 0.2590  data_time: 0.0224  time: 0.0578
```

By analyzing the COCO Eval results, we can identify issues and take corresponding measures for optimization. The optimization direction is suggested to start with the dataset, followed by training parameters, and then the model structure.

Average Precision (AP):
- At IoU=0.50:0.95 and area=all, AP is 0.254, which is at a medium-low level overall. The model has room for improvement in detection accuracy under different intersection-over-union ratios.
- When IoU=0.50, AP reaches 0.743, indicating that the model can perform well under loose intersection-over-union requirements. However, at IoU=0.75, AP is only 0.093, meaning the model performs poorly under high intersection-over-union requirements, especially when the prediction box and the ground truth box need to coincide closely.
- Classified by detection target area, area=small has an AP of -1.000, indicating a severe problem with small target detection, and the validation set lacks small targets. Area=medium has an AR and AP of 0, indicating that there are some other issues, such as a lack of medium targets in the training set or abnormal data augmentation parameters.

Average Recall (AR):
- At IoU=0.50:0.95 and area=all under different maxDets, as maxDets increases from 1 to 100, AR increases from 0.331 to 0.453. Increasing the maximum number of detectable targets can improve recall to some extent, but the overall values are not high, and the model may miss many targets in actual situations.
- In the area classification, area=small has an AR of -1.000, again highlighting the issue of lacking small targets in the validation set.

Based on the above data, we first check whether there are enough small targets in the dataset, whether the data annotation for small targets is accurate and complete, and if necessary, re-annotate to ensure that the annotation box fits the actual boundary of small targets. Then, check the dataset after it has passed through the training pipeline, and ensure that the image colors and annotations after data augmentation are correct and reasonable.

In addition, we also need to check the training process, whether the model has converged, etc. You can use Tensorboard to view this.

Install and run Tensorboard:

```sh
python3 -m pip install tensorboard && \
    tensorboard --logdir workdir
```

Under the Scalars tab, you can view the changes of recorded scalar metrics (such as loss, accuracy) over time (usually training epochs). By observing the downward trend of the loss function and the upward trend of accuracy, you can judge whether the model is converging normally. If the loss function no longer decreases or the accuracy no longer increases, it may indicate that the model has converged or there is a problem. Here, we only briefly introduce the adjustment strategy.

- **Learning Rate**: If the loss function decreases too slowly, you can try increasing the learning rate; if the loss function shows violent fluctuations or does not converge, it may be that the learning rate is too large, and you need to reduce the learning rate. For adjustment strategies for the learning rate, please refer to [SSCMA - Customization - Basic Configuration Structure](../../custom/basics.md).
- **Number of Iterations**: If the model has not fully converged during training (for example, the loss function is still decreasing, and accuracy is still increasing), you can appropriately increase the number of iterations. If the model has already converged, continuing to increase the number of iterations may lead to overfitting, in which case you can reduce the number of iterations.

:::

Find the trained model in the `work_dirs/rtmdet_nano_8xb32_300e_coco` directory. In addition, when the model training result accuracy is poor, you can analyze the COCO Eval results to find the problem and take corresponding measures for optimization.

:::tip

When the model training result accuracy is poor, you can analyze the COCO Eval results to find the problem and take corresponding measures for optimization.

:::


## Model Exporting and Verification

During the training process, you can view the training logs, export the model, and verify the model's performance at any time. Some of the metrics output during model verification are also displayed during training, so in this part, we will first introduce how to export the model and then discuss how to verify the accuracy of the exported model.

### Exporting the Model

Here, we take exporting the TFLite model as an example. You can use the following command to export TFLite models of different accuracies:

```sh
python3 tools/export.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    work_dirs/epoch_50.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    --imgsz 192 192 \
    --format tflite
```

:::warning

We recommend using the same resolution for training and exporting. Using different resolutions for training and exporting may result in reduced model accuracy or complete loss of accuracy.

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

### Model Verification

After exporting, you can use the following command to verify the TFLite Int8 model:

```sh
python3 tools/test.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    work_dirs/epoch_50_int8.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    imgsz='(192,192)'
```

You will get the following output:

```sh
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.163
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.064
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.409
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
```

From the verification results, it can be seen that the exported model's performance on the verification set is different from its performance during training, with a decrease of 8.1% in AP@50:95 and a decrease of 33.8% in AP@50. You can try using QAT to reduce the loss of quantization accuracy.

:::tip

For a detailed explanation of the above output, please refer to [COCO Dataset Evaluation Metrics](https://cocodataset.org/#detection-eval), where we mainly focus on mAP at 50-95 IoU and 50 IoU.

:::


### QAT

QAT (Quantization-Aware Training) is a method that simulates quantization operations during the model training process, allowing the model to gradually adapt to quantization errors, thereby maintaining higher accuracy after quantization. SSCMA supports QAT, and you can refer to the following method to obtain a QAT model and verify it again.

```sh
python3 tools/quantization.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    work_dirs/epoch_50.pth \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    imgsz='(192,192)' \
    max_epochs=50
```

After QAT training is completed, the quantized model will be automatically exported, and its storage path will be `out/qat_model_test.tflite`. You can use the following command to verify it:

```sh
python3 tools/test.py \
    configs/rtmdet_nano_8xb32_300e_coco.py \
    out/qat_model_test.tflite \
    --cfg-options \
    data_root=$(pwd)/datasets/coco_mask/mask/ \
    num_classes=2 \
    train_ann_file=train/_annotations.coco.json \
    val_ann_file=valid/_annotations.coco.json \
    train_img_prefix=train/ \
    val_img_prefix=valid/ \
    imgsz='(192,192)'
```
