# onnx转至ncnn及量化

为了在边缘设备上运行以训练好的模型，需要的内存和算力都有一定的限制。一般为了能够在这些是被上运行会将训练好的模型进行8bit量化，不仅能够加速推理(在某些硬件架构上)同时也能极大的减少运行内存。

## 准备

1. 在将模型导出到ncnn或量化前，需要有编译好的ncnn工具链。
2. 准备好训练好的模型(onnx格式)。
3. 准备校验数据，如果需要ncnn量化或者onnx静态量化需要准备校验数据。

若您的系统为ubuntu可直接运行[tools/env_config.py](../../../tools/env_config.py)
脚本，将自动编译安装ncnn，导出onnx模型可在[tools/torch2onnx.py](../../../tools/torch2onnx.py)中实现，导出onnx如下所示:

```shell
python ./tools/torch2onnx.py  --config $CONFIG_PATH --checkpoint $PTH_PATH  --shape $IMGSZ --task $TASK
```

##### 参数解释

- `$CONFIG_PATH` 模型的配置文件路径。
- `$PTH_PATH` 模型权重文件路径。
- `$IMGSZ` 模型输入数据形状，例如 `112`或者`128,96`。
- `$TASK` 模型所属任务,可在`mmdet`、`mmcls`、`mmpose`中选一个，分别表示目标检测、分类、关键点检测任务。

## 示例

- 以下为将resnet50模型(onnx格式)转ncnn及量化示例。其主要通过[tools/export_quantize.py](../../../tools/export_quantize.py)脚本完成，
- 本示例resnet50.onnx模型文件位于项目根目录下，校验数据文件夹位于根目录下的`./images`文件夹下。

**提示：** 所导出的所有文件位于所输入onnx文件的同级目录下，导出的文件可通过[netron](https://netron.app)查看。

1. onnx float32转float16

可通过如下命令完成：

```shell
python tools/export_quantize.py ./resnet50.onnx  --type onnx_fp16
```

完成后会在原resnet50.onnx同级目录下产生转换后的文件resnet50_fp16.onnx目标文件

2. onnx 8bit量化

- 脚本支持onnx的动态量化和静态量化。\
**注意：** 在静态量化时需要提供一个校验的数据文件路径

```shell
# 动态量化 不需要提供校验数据
python tools/export_quantize.py ./resnet50.onnx --type onnx_quan_dy
#静态量化 需要提供校准数据集文件路径
python tools/export_quantize.py ./resnet50.onnx --images ./images --type onnx_quan_st
# 可同时进行动态量化和静态量化
python tools/export_quantize.py ./resnet50.onnx --images ./images --type onnx_quan_dy onnx_quan_st
```

完成后会在resnet50.onnx同级文件下产生resnet50_quan_dynamic.onnx  或 resnet50_quan_static.onnx的目标文件

3. 导出至ncnn

```shell
python tools/export_quantize.py ./resnet50.onnx --type ncnn
```

完成后会在resnet50.onnx同级文件下产生resnet50.bin 和resnet50.param文件

4. ncnn float32 转 float16

- 这一步程序会将onnx文件转至ncnn float32后再将ncnn float32 转ncnn float16

```shell
python tools/export_quantize.py ./resnet50.onnx --type ncnn_fp16
```

完成后会在resnet50.onnx同级文件下产生resnet50_fp16.bin 和resnet50_fp16.param文件

5. ncnn量化至8bit

- 同样，这一步的前提是需要有ncnn的.param文件和.bin文件方可进行，如果没有相应文件程序会自动生成。

```shell
python tools/export_quantize.py ./resnet50.onnx --images ./images --type ncnn_quan
```

完成后会在resnet50.onnx同级文件下产生resnet50_int8.bin 和resnet50_int8.param文件

## 参数解释

打开[tools/export_quantize.py](../../../tools/export_quantize.py)文件即可发现，脚本一共有三个参数onnx、images、type。

- `--onnx` 为所需要的onnx模型文件路径
- `--images` 为校验数据集文件夹路径
- `--type` 为所导出模型类型的参数，需在`onnx_fp16`, `onnx_quan_st`, `onnx_quan_dy`, `ncnn`, `ncnn_fp16`, `ncnn_quan`中选择一个或多个，分别表示onnx fp16量化、onnx静态量化、onnx动态量化、onnx转ncnn、ncnn fp16量化、ncnn 8bit量化。
