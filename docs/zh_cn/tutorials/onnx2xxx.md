# onnx转至ncnn及量化

为了在边缘设备上运行以训练好的模型，需要的内存和算力都有一定的限制。一般为了能够在这些是被上运行会将训练好的模型进行8bit量化，不仅能够加速推理(在某些硬件架构上)同时也能极大的减少运行内存。

## 准备

在将模型导出到ncnn或量化前，需要有编译好的ncnn工具链，同时准备好训练好的模型(onnx格式)。若您的系统上为满足以上条件可运行[tools/env_config.py](../../../tools/env_config.py)
脚本编译安装ncnn，导出onnx模型可在[tools/torch2onnx.py](../../../tools/torch2onnx.py)中实现。

## 示例

以下为将resnet50模型(onnx格式)转ncnn及量化示例。其主要通过[tools/export_quantize.py](../../../tools/export_quantize.py)脚本完成，
本示例resnet50.onnx模型文件位于项目根目录下。\
**提示：** 所导出的所有文件位于所输入onnx文件的同级目录下，导出的文件可通过[netron](https://netron.app)查看。

1. onnx float32转float16

可通过如下命令

```shell
python tools/export_quantize.py ./resnet50.onnx  --type onnx_fp16
```

之后会在原onnx模型文件同级目录下产生转换后的文件resnet50_fp16.onnx

2. onnx 8bit量化

脚本支持onnx的动态量化和静态量化。\
**注意：** 在静态量化时需要提供一个校验的数据文件路径

```shell
# 动态量化
python tools/export_quantize.py ./resnet50.onnx --type onnx_quan_dy
#静态量化 需要提供校准数据集文件路径
python tools/export_quantize.py ./resnet50.onnx --images-path ./images --type onnx_quan_st
# 可同时进行动态量化和静态量化
python tools/export_quantize.py ./resnet50.onnx --images-path ./image --type onnx_quan_dy onnx_quan_st
```

3. 导出至ncnn

```shell
python tools/export_quantize.py ./resnet50.onnx --type ncnn
```

4. ncnn float32 转 float16 \
   这一步程序会将onnx文件转至ncnn float32后再将ncnn float32 转ncnn float16

```shell
python tools/export_quantize.py ./resnet50.onnx --type ncnn_fp16
```

5. ncnn量化至8bit
   同样，这一步的前提是需要有ncnn的.param文件和.bin文件方可进行

```shell
python tools/export_quantize.py ./resnet50.onnx --images ./images --type ncnn_quan
```

## 参数解释

打开[tools/export_quantize.py](../../../tools/export_quantize.py)文件即可发现，脚本一共有三个参数onnx、imags-path、type。

- `--onnx` 为所导出的onnx模型文件\
- `--imags-path` 为校验数据集文件夹路径\
- `--type` 为所导出模型类型的参数，需在`onnx_fp16`, `onnx_quan_st`, `onnx_quan_dy`, `ncnn`, `ncnn_fp16`, `ncnn_quan`中选择。
 