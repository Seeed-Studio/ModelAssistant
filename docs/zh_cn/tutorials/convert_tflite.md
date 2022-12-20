# torch模型转换为TFLite

将torch训练得到的模型权重从float32量化到int8，从而减少内存，降低算力的要求，模型因此可以应用在低功耗嵌入式设备上。
而当前主流量化方法是TFLite，针对本仓库支持的模型，提供转化的方法流程。

## 准备

1. 确保已经得到训练好的模型权重。
2. 转换TFLite需要代表数据集，请使用训练数据集或者准备一个与训练数据相似的标准数据集(100数据)，推荐直接使用训练数据集。
请务必保证所使用的的代表数据集与训练数据相似。

### Python
```shell
python ./tool/export.py $CONFIGS --weights $WEIGHTS_PATH --data_root $REPRESENTATIVE_DATASET --types $MODEL_TYPE --shape $INPUT_SHAPE --classes $AUDIO_CLASSES --fp16 $FP16
```

##### 参数说明
- `$CONFIGS` 模型对应配置文件(configs目录下)。
- `$WEIGHTS_PATH` torch模型权重的文件路径。
- `$REPRESENTATIVE_DATASET` 代表数据集文件目录的路径，推荐使用训练数据集。
- `$MODEL_TYPE` 代表数据集的数据类型，1对应一维音频数据集， 2对应二维图片数据集, 默认是2。
- `$INPUT_SHAPE` 输入数据的形状，默认pfld模型：'112'或'112 112', audio模型：'8192'。
- `$AUDIO_CLASSES` audio模型的输出类别数，只针对audio模型，默认：'4'。
- `$FP16` 是否转化为fp16的tflite模型，默认是否，添加参数表示为是。

## 示例

- 将pfld模型(pfld.pth)从torch转换为tflite。 假定：代表数据集文件目录(pfld_data)位于根目录下，
torch模型权重位于根目录下，输入图片大小设置为112，输出int8模型。

**提示：** 所导出的TFLite文件位于torch模型权重的同级目录下, 若要输出fp16的tflite模型，则需要添加--fp16参数。
audio模型需要根据输出类别数判断是否添加--classes参数，且需要添加--type参数，设置为1。

### Python
```shell
python ./tool/export.py configs/pfld/pfld_mv2n_112.py --weights pfld.pth --data_root pfld_data --shape 112
```
导出成功会显示相应的tflite保存路径。