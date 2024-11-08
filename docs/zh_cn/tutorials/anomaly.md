# 异常检测训练及测试

本节将介绍如何使用异常检测，内容包括以下：

1. 数据集采集
2. 数据预处理
3. 模型训练
4. 模型导出
5. 测试 

# 数据集采集

```bash
python tools/dataset_tool/read_serial.py -sr ${采样率} -p ${端口号} -f ${保存csv文件路径} 
```

通过CTL+C取消采集，在数据采集完成后会再指定路径下生成一个csv文件。

# 数据预处理

```bash
python tools/dataset_tool/signal_data_processing.py -t ${数据标签} -f ${数据文件路径} -sr ${采样率} 
```
执行完毕后会在当前目录下建立一个`dataset`文件夹，文件夹内会有相应数据标签的数据
其文件结构如下
```bash
dataset
    |
    |--train
    |    |--xxx.npy
    |    |--xxx.npy
    |
    |----val
         |--xxx.npy
         |--xxx.npy
```

# 模型训练

```bash
python tools/train.py configs/anomaly/vae_mirophone.py --cfg-options data_root=${数据集路径}
```
训练完成后默认会在work_dirs路径下生成相应的模型权重文件。


# 模型导出
```bash
python tools/export.py configs/anomaly/vae_mirophone.py work_dirs/epoch_100.pth --imgsz 32 32
```
导出的模型与模型在相同路径，相同文件名，但是不同后缀。

# 测试

```bash
python tools/test.py configs/anomaly/vae_mirophone.py work_dirs/epoch_100.onnx 
```
测试完成后将会显示模型的mse结果，此结果应与训练模型结果对齐。
