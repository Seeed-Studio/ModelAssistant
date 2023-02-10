# Tutorial 5: Pytorch 到 TFLite 的模型转换（实验性支持）
- [Tutorial 5: Pytorch 到 TFLite 的模型转换（实验性支持）](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#tutorial-5-pytorch-%E5%88%B0-tflite-%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2%E5%AE%9E%E9%AA%8C%E6%80%A7%E6%94%AF%E6%8C%81)
    - [pytorch模型如何转换到TFLite](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#tutorial-5-pytorch-%E5%88%B0-tflite-%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%BD%AC%E6%8D%A2%E5%AE%9E%E9%AA%8C%E6%80%A7%E6%94%AF%E6%8C%81-1)
        - [准备](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#%E5%87%86%E5%A4%87)
        - [使用](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#%E4%BD%BF%E7%94%A8)
        - [参数描述](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#%E5%8F%82%E6%95%B0%E6%8F%8F%E8%BF%B0)
    - [转换后的模型如何验证](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#%E8%BD%AC%E6%8D%A2%E5%90%8E%E7%9A%84%E6%A8%A1%E5%9E%8B%E5%A6%82%E4%BD%95%E9%AA%8C%E8%AF%81)
        - [准备](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#%E5%87%86%E5%A4%87-1)
        - [使用](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#%E4%BD%BF%E7%94%A8-1)
        - [参数描述](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#%E5%8F%82%E6%95%B0%E6%8F%8F%E8%BF%B0-1)
        - [模型和结果](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#%E6%A8%A1%E5%9E%8B%E5%92%8C%E7%BB%93%E6%9E%9C)
    - [提醒](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#%E6%8F%90%E9%86%92)
    - [FAQs](https://github.com/Seeed-Studio/EdgeLab/blob/torch2tflite/docs/zh_CN/tutorials/pytorch2tflite.md#faqs)

## pytorch模型如何转换到TFLite
---
### 准备
1. 确保已经按照[安装指导](https://github.com/Seeed-Studio/EdgeLab/blob/master/docs/en/get_started/installation.md)安装好所有依赖包.
2. 确保已经准备好torch模型，如果没有，可以按照[模型训练](https://github.com/Seeed-Studio/EdgeLab/blob/new-catalog/docs/zh_CN/tutorials/tranning.md)训练一个模型，或者从[model_zoo](https://github.com/Seeed-Studio/EdgeLab/releases/tag/model_zoo)下载所需要的模型。
3. TFLite模型导出需要训练集作为代表数据集，如果没有找到，程序会自动下载。但对于某些大数据集，这会花费很长时间，请耐心等待。

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
    python tools/torch2tflite/py \
        mmdet \
        configs/fomo/fomo_mobnetv2_0.35_x8_abl_coco.py \
        --weights fomo_model.pth \
        --tflite_type int8 \
        --cfg-options data_root=/home/users/datasets/fomo.v1i.coco/ \
#### pfld模型:
    python tools/torch2tflite/py \
        mmpose \
        cconfigs/pfld/pfld_mv2n_112.py \
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