# Pytorch 到 ONNX 的模型转换（实验性支持）
- [Pytorch到ONNX（实验性支持）](#pytorch-到-onnx-的模型转换实验性支持)
    - [pytorch模型如何转换到onnx](#pytorch模型如何转换到onnx)
        - [准备](#准备)
        - [使用](#使用)
        - [参数描述](#参数描述)
    - [转换后的模型如何验证](#转换后的模型如何验证)
        - [准备](#准备-1)
        - [使用](#使用-1)
        - [参数描述](#参数描述-1)
        - [模型和结果](#模型和结果)
    - [提醒](#提醒)
    - [FAQs](#faqs)

## Pytorch模型如何转换到onnx

### 准备
1. 确保已经按照[安装指导](../../get_started/installation.md)安装好所有依赖包.
2. 安装推理所需要的库. 使用下面的命令:
    ```
    pip install -r requirements/inference.txt
    ```
3. 确保已经准备好torch模型，如果没有，可以按照[模型训练](../training/index.rst)训练一个模型，或者从[model_zoo](https://github.com/Seeed-Studio/EdgeLab/releases/tag/model_zoo)下载所需要的模型。

### 使用
    python tools/torch2onnx.py \
        ${TYPE} \
        ${CONFIG_FILE} \
        --checkpoint ${CHECKPOINT_FILE} \
        --simplify ${SIMPLIFY} \
        --shape ${DATA_SHAPE} \
        --audio {AUDIO_FLAG} \

### 参数描述
- `${TYPE}` 训练模型的类型，[`mmdet`, `mmcls`, `mmpose`]。
- `${CONFIG_FILE}` 模型配置文件(配置路径下)。
- `--checkpoint` torch模型路径。
- `--simplify` 是否简化onnx模型，如果给定会设置为**True**。
- `--shape` 输入数据的形状。
- `--audio` 如果给定，会选择音频数据加载方式。

#### 例子:
#### fomo模型:
    python tools/torch2onnx.py \
        mmdet \
        configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
        --checkpoint fomo_model.pth \
        --shape 96 \
#### pfld模型:
    python tools/torch2onnx.py \
        mmpose \
        configs/pfld/pfld_mv2n_112.py \
        --checkpoint pfld_mv2n_112.pth \
        --shape 112 \
 

## 转换后的模型如何验证

可以使用 `tools/test.py` 去验证onnx模型。

### 准备

测试数据集已经在每个模型对应的配置文件中设置，如果需要测试自定义数据集，请参照[custom dataset(TODO)](../datasets/index.rst)。

### 使用
    python tools/test.py \
    ${TYPE} \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    --audio ${AUDIO_FLAG} \
    --out ${OUTPUT_FILE} \
    --data ${DATA_ROOT} \
    --no-show ${SHOW_RESULT} \
    --cfg-options ${CFG-OPTIONS} \

### 参数描述
- `${TYPE}` 训练数据类型，[`mmdet`, `mmcls`, `mmpose`]。
- `${CONFIG_FILE}` 模型配置文件(配置路径下)。
- `${CHECKPOINT_FILE}` TFLite模型路径。
- `--audio` 如果给定，会选择音频数据加载方式。
- `--out` 输出结果文件保存路径。
- `--data` 手动指定数据根目录。
- `--no-show` 是否显示结果图片, 如果没有指定会设置为 **False**并显示图片。
- `--cfg-options` 在配置文件中重写一些配置参数, xxx=yyy的键值对格式会重写到配置文件中。

#### 例子:
#### fomo模型（待定）:

#### pfld模型:
    python tools/test.py \
    mmpose \
    configs/pfld/pfld_mv2n_112.py \
    pfld_mv2n_112.onnx \
    --no_show \

### 模型和结果

| Model |           Config               |   Acc  |
| :--: | :--: |:--:|
| pfld  | configs/pfld/pfld_mv2n_112.py  |   98.77% |
| fomo  | configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py |     |


### 提醒
- 


### FAQS
- 