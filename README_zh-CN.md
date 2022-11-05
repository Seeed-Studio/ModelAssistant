# 基于 OpenMMLab 框架的多任务网络模型库

[English](./README.md) | 简体中文

## 简介

这是一个基于 [OpenMMLab]() 框架开发的一个适用于多种视觉任务主干网络的模型训练库。\
同时可以使用[MMDetection]()、[MMClassification]()、[MMPose]()等工具箱的模型并训练、测试和导出。\
通过融合后的训练脚本我们可以很轻松地训练一个以上工具箱中已有的模型，只需要修改配置文件的数据集路径即可训练自定义数据集。、

## 相关教程和文档

可查看[快速使用文档](./docs/zh_cn/get_started.md)学习本项目的基本使用，同时也可使用我们提供的[colab教程]()。\
对于更多教程可以查看以下内容：

1. [基础环境配置](./docs/zh_cn/tutorials/env_config.md)
2. [自定义数据集配置](./docs/zh_cn/tutorials/datasets_config.md)
3. [Pytorch模型转至ONNX](./docs/zh_cn/tutorials/pytorch2onnx.md)
4. [ONNX转NCNN及量化](./docs/zh_cn/tutorials/onnx2xxx.md)
5. [相关工具的使用](./docs/zh_cn/tutorials/use_tools.md)

## 模型库

对于mmdetection、mmpos、mmclassify所支持的模型均可使用。
本项目直接实现的算法有以下：\
[PFLD](./configs/pfld/README.md)\
[EAT](./configs/audio_classify/README.md)

## FAQ

对于在环境配置与训练过程中可能出现的问题可先查看[相关问题解决文档](./docs/zh_cn/faq.md)
查看。若没能解决您的问题可提出[issue](https://github.com/Seeed-Studio/edgelab/issues)，
我们会尽快为您解决。

## 许可证

edgelab 目前以 Apache 2.0 的许可证发布，但是其中有一部分功能并不是使用的 Apache2.0 许可证，我们在[许可证](./LICENSES.md)
中详细地列出了这些功能以及他们对应的许可证，如果您正在从事盈利性活动，请谨慎参考此文档。

