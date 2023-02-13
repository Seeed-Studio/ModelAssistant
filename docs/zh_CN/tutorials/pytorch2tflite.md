# Tutorial 5: Pytorch 到 TFLite 的模型转换（实验性支持）
- [Tutorial 5: Pytorch 到 TFLite 的模型转换（实验性支持）](#tutorial-5-pytorch-到-tflite-的模型转换实验性支持)
    - [pytorch模型如何转换到TFLite](#pytorch模型如何转换到tflite)
        - [准备](#准备)
        - [使用](#使用)
        - [参数描述](#参数描述)
    - [转换后的模型如何验证](#转换后的模型如何验证)
        - [准备](#e58786e5a487-1)
        - [使用](#e4bdbfe794a8-1)
        - [参数描述](#e58f82e695b0e68f8fe8bfb0-1)
        - [模型和结果](#模型和结果)
    - [提醒](#提醒)
    - [FAQs](#faqs)

## pytorch模型如何转换到TFLite
---
### 准备
1. 确保已经按照[安装指导](https://github.com/Seeed-Studio/EdgeLab/blob/master/docs/zh_CN/get_started/installation.md)安装好所有依赖包.
2. 安装推理所需要的库. 使用下面的命令:
    ```
    pip install -r requirements/inference.txt
    ```
3. 确保已经准备好torch模型，如果没有，可以按照[模型训练](https://github.com/Seeed-Studio/EdgeLab/blob/master/docs/zh_CN/tutorials/trainning.md)训练一个模型，或者从[model_zoo](https://github.com/Seeed-Studio/EdgeLab/releases/tag/model_zoo)下载所需要的模型。
4. TFLite模型导出需要训练集作为代表数据集，如果没有找到，程序会自动下载。但对于某些大数据集，这会花费很长时间，请耐心等待。

### 使用
    python tools/torch2tflite.py \
        ${TYPE} \
        ${CONFIG_FILE} \
        --weights ${CHECKPOINT_FILE} \
        --tflite_type ${TFLITE_TYPE} \
        --cfg-options ${CFG_OPTIONS} \
        --audio {AUDIO_FLAG} \

### 参数描述
- `${TYPE}` 训练模型的类型，[`mmdet`, `mmcls`, `mmpose`]。
- `${CONFIG_FILE}` 模型配置文件(配置路径下)。
- `--weights` torch模型路径。
- `--tflite_type` TFLite量化类型, `int8`, `fp16`, `fp32`, 默认: `int8`。
- `--cfg-options`: 在配置文件中重写一些配置参数, xxx=yyy的键值对格式会重写到配置文件中。
- `--audio` 如果给定，会选择音频数据加载方式。

#### 例子:
#### fomo模型:
    python tools/torch2tflite.py \
        mmdet \
        configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
        --weights fomo_model.pth \
        --tflite_type int8 \
        --cfg-options data_root=/home/users/datasets/fomo.v1i.coco/ \
#### pfld模型:
    python tools/torch2tflite.py \
        mmpose \
        configs/pfld/pfld_mv2n_112.py \
        --weights pfld_mv2n_112.pth \
        --tflite_type int8 \

**注意：** TFLite模型保存路径与torch模型路径相同，对于fomo模型，data_root并未在配置文件中给定，需要手动配置。

## 转换后的模型如何验证
---
可以使用[tools/test.py](https://github.com/Seeed-Studio/EdgeLab/blob/master/tools/test.py)去验证TFLite模型。

### 准备
- 测试数据集已经在每个模型对应的配置文件中设置，如果需要测试自定义数据集，请参照[custom dataset(TODO)]()。

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
    pfld_mv2n_112.pth \
    --no_show \

### 模型和结果

| Model |           Config               |   Acc  |
| :--: | :--: |:--:|
| pfld  | configs/pfld/pfld_mv2n_112.py  |   98.76% |
| fomo  | configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py |     |


### 提醒
- 


### FAQS
- 