# Tutorial 5: Pytorch To TFLite (Experimental)
- [Tutorial 5: Pytorch To TFlite (Experimental)](#tutorial-5-pytorch-to-tflite-experimental)
    - [How to convert models from Pytorch to TFLite](#how-to-convert-models-from-pytorch-to-tflite)
        - [Preparation](#preparation)
        - [Usage](#usage)
        - [Description of all arguments](#description-of-all-arguments)
    - [How to evaluate the exported models](#how-to-evaluate-the-exported-models)
        - [Preparation](#preparation-1)
        - [Usage](#usage-1)
        - [Description of all arguments](#description-of-all-arguments-1)
        - [Results and Models](#results-and-models)
    - [Reminders](#reminders)
    - [FAQs](#faqs)

## How to convert models from Pytorch to TFLite
---
### Preparation
1. Make sure you have installed all packages following [get_started/installation.md](https://github.com/Seeed-Studio/EdgeLab/blob/master/docs/en/get_started/installation.md).
2. Install libraries needed for inference. Using command as below:
    ```
    pip install -r requirements/inference.txt
    ```
3. Make sure the torch model is ready, if not, you can train a model follow [tutorials/trainning.md](https://github.com/Seeed-Studio/EdgeLab/blob/master/docs/en/tutorials/trainning.md) or download from [model_zoo](https://github.com/Seeed-Studio/EdgeLab/releases/tag/model_zoo).
4. Export TFLite model requires the training dataset as a representative dataset, which can be download automatically if not have. But for some large datasets, it will take a lot of time, please wait.

### Usage
    python tools/torch2tflite.py \
        ${TYPE} \
        ${CONFIG_FILE} \
        --weights ${CHECKPOINT_FILE} \
        --tflite_type ${TFLITE_TYPE} \
        --cfg-options ${CFG_OPTIONS} \
        --audio {AUDIO_FLAG} \

### Description of all arguments
- `${TYPE}` Type for training model，[`mmdet`, `mmcls`, `mmpose`]。
- `${CONFIG_FILE}` Configuration file for model(under the configs directory).
- `--weights` Path of torch model.
- `--tflite_type` Quantization type for tflite, `int8`, `fp16`, `fp32`, default: `int8`.
- `--cfg-options`: Override some settings in the used config file, the key-value pair in xxx=yyy format will be merged into config file.
- `--audio` Choose audio dataset load code if given.

#### Example:
#### fomo model:
    python tools/torch2tflite.py \
        mmdet \
        configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
        --weights fomo_model.pth \
        --tflite_type int8 \
        --cfg-options data_root=/home/users/datasets/fomo.v1i.coco/ \
#### pfld model:
    python tools/torch2tflite.py \
        mmpose \
        configs/pfld/pfld_mv2n_112.py \
        --weights pfld_mv2n_112.pth \
        --tflite_type int8 \

**Note：** TFLite model is saved in the same path as torch model. Data_root for fomo is not given in configuration file, please set it manually.  

## How to evaluate the exported models
---
You can use [tools/test.py](https://github.com/Seeed-Studio/EdgeLab/blob/master/tools/test.py) to evaluate TFLite model.

### Preparation
- Test dataset is set from the corresponding [config file](https://github.com/Seeed-Studio/EdgeLab/tree/master/configs) for each model. If you want test the custom dataset, please follow [custom dataset(TODO)]().

### Usage
    python tools/test.py \
    ${TYPE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --audio ${AUDIO_FLAG} \
    --out ${OUTPUT_FILE} \
    --data ${DATA_ROOT} \
    --no-show ${SHOW_RESULT} \
    --cfg-options ${CFG-OPTIONS} \

### Description of all arguments
- `${TYPE}` Type for training model，[`mmdet`, `mmcls`, `mmpose`]。
- `${CONFIG_FILE}` Configuration file for model(under the configs directory).
- `${CHECKPOINT_FILE}` Path to TFLite model.
- `--audio` Choose audio dataset load code if given.
- `--out` The path of output result file.
- `--data` Specify data root manually.
- `--no-show` Show result image or not, If not specified, it will be set to **False**, then show result image.
- `--cfg-options` Override some settings in the used config file, the key-value pair in xxx=yyy format will be merged into config file.

#### Example:
#### fomo model(TODO):

#### pfld model:
    python tools/test.py \
    mmpose \
    configs/pfld/pfld_mv2n_112.py \
    pfld_mv2n_112.pth \
    --no_show \

### Results and Models

| Model |           Config               |   Acc  |
| :--: | :--: |:--:|
| pfld  | configs/pfld/pfld_mv2n_112.py  |   98.76% |
| fomo  | configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py |     |


### Reminders
- None


### FAQS
- None