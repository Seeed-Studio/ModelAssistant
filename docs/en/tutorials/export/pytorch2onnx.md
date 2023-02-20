# Pytorch To ONNX (Experimental)
- [Pytorch To ONNX (Experimental)](#pytorch-to-onnx-experimental)
    - [How to convert models from Pytorch to ONNX](#how-to-convert-models-from-pytorch-to-onnx)
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

## How to convert models from Pytorch to ONNX

### Preparation
1. Make sure you have installed all packages following [get_started/installation.md](../../get_started/installation.md).
2. Install libraries needed for inference. Using command as below:
    ```
    pip install -r requirements/inference.txt
    ```
3. Make sure the torch model is ready, if not, you can train a model follow [tutorials/trainning.md](../training/index.rst) or download from [model_zoo](https://github.com/Seeed-Studio/EdgeLab/releases/tag/model_zoo).

### Usage
```sh
python tools/torch2onnx.py \
    ${TYPE} \
    ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT_FILE} \
    --simplify ${SIMPLIFY} \
    --shape ${DATA_SHAPE} \
    --audio ${AUDIO_FLAG} \
```

### Description of all arguments
- `${TYPE}` Type for training model，[`mmdet`, `mmcls`, `mmpose`]。
- `${CONFIG_FILE}` Configuration file for model(under the configs directory).
- `--checkpoint` Path of torch model.
- `--simplify` Whether to simplify onnx model, it will be **True** if given.
- `--shape` Input data size.
- `--audio` Choose audio dataset load code if given.

#### Example:
#### fomo model:
```sh
python tools/torch2onnx.py \
    mmdet \
    configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
    --checkpoint fomo_model.pth \
    --shape 96 \
```

#### pfld model:
```sh
python tools/torch2onnx.py \
    mmpose \
    configs/pfld/pfld_mv2n_112.py \
    --checkpoint pfld_mv2n_112.pth \
    --shape 112 \
```

## How to evaluate the exported models

You can use `tools/test.py` to evaluate ONNX model.

### Preparation

Test dataset is set from the corresponding [config file](../config.md) for each model. If you want test the custom dataset, please follow [custom dataset(TODO)](../datasets/index.rst).

### Usage
```sh
python tools/test.py \
    ${TYPE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --audio ${AUDIO_FLAG} \
    --out ${OUTPUT_FILE} \
    --data ${DATA_ROOT} \
    --no-show ${SHOW_RESULT} \
    --cfg-options ${CFG-OPTIONS} \
```

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
```sh
python tools/test.py \
    mmpose \
    configs/pfld/pfld_mv2n_112.py \
    pfld_mv2n_112.onnx \
    --no_show \
```

### Results and Models

| Model |           Config               |   Acc  |
| :--: | :--: |:--:|
| pfld  | configs/pfld/pfld_mv2n_112.py  |   98.77% |
| fomo  | configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py |     |


### Reminders
- None


### FAQS
- None