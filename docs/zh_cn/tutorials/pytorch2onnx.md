# Pytorch模型转ONNX

本项目的模型转换脚本位于tools文件夹下的[torch2onnx.py](../../../tools/torch2onnx.py)文件，可分别转换检测、分类、回归模型

## 1.参数解读

1. `--config` 所要转换模型的训练配置文件
2. `--checkpoint` 模型训练完成后生成的模型权重文件(`.pth`文件)
3. `--task` 模型的任务类型,`mmdet`、`mmcls`、`mmpose`可选，分别对应目标检测、分类、关键点检测。
4. `--input-img` 在转换检测模型时必须提供的参数，用于模型转换时进行推理
5. `--show` 在转换模型时是否显示转换信息，同时会将节点的相关信息放入onnx模型中,建议设置为False，可减少输出模型文件大小。
6. `--verify` 在模型转换成功后是否验证原模型(pytorch模型)和转换后的onnx模型的推理结果是否相同。
7. `--output-file` 导出onnx模型的文件名，同时导出的模型文件位于和原模型权重文件想通的文件夹下。
8. `--opset-version` 所导出的onnx模型的算子集版本，
9. `--simplify` 是否使用onnx简化工具，可消除部分胶水算子
10. `--shape` 模型输入数据的大小，若输入图像则为图像的宽高，若为音频数据则为数据的长度。如果只有一个整数且输入数据为图像时，则代表其宽高都等于此值。
11. `--audio` 模型输入是否为音频数据，用于确定模型输入形状的确定。
12. `--dynamic-export` 导出的模型是否需要使用动态维度，使用动态维度时，其BATCH，WEIDTH，HEIGHT值可不固定。

## 2.目标检测模型的转换

检测模型可使用一下命令转换已经训练好的模型。

```shell
python ./tools/torch2onnx.py --config $CONFIG --checkpoint $CHECKPOINT --task mmdet --input_img $IMAG_PATH --shape $shape --simplify --output-file $OUTPUT_FILE
```

## 3.分类模型的转换

分类模型与检测模型转换命令相差不大，需要将`--task`参数改为`mmcls` 和删除`--input-img`参数,其命令如下：

```shell
python ./tools/torch2onnx.py --config $CONFIG --checkpoint $CHECKPOINT --task mmdcls ---shape $shape --simplify --output-file $OUTPUT_FILE
```

若输入数据有一维音频数据，则需在命令上添加`--audio`参数,如下：

```shell
python ./tools/torch2onnx.py --config $CONFIG --checkpoint $CHECKPOINT --task mmdcls ---shape $shape --simplify --output-file $OUTPUT_FILE --audio
```

## 4.关键点检测模型的转换

对于关键点检测模型的转换，其命令如下：

```shell
python ./tools/torch2onnx.py --config $CONFIG --checkpoint $CHECKPOINT --task mmdpose ---shape $shape --simplify --output-file $OUTPUT_FILE
```
