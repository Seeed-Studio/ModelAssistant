# 训练与验证管线

在训练与验证过程中，我们需要对数据进行预处理、特征工程、模型训练、模型评估等一系列操作。为了方便用户进行这些操作，我们提供了一套完整的训练与验证管线，用户可以通过简单的配置来完成整个训练与验证过程。

:::tip

在训练流程中，数据集和数据加载器是核心组件，SSCMA 基于 MMEngine，因此这方面与 PyTorch 中的概念相同。数据集负责定义数据量、读取和预处理方式，而数据加载器则负责按批次大小、随机乱序和并行等设置迭代加载数据，形成数据源。更多请参考 [PyTorch 数据加载](https://pytorch.org/docs/stable/data.html)。

:::


在 MMEngine 的执行器（Runner）中，你可以通过设置三个特定的参数来指定数据加载器，以便在不同的训练阶段使用：

- train_dataloader：此参数在执行器的 Runner.train() 方法中调用，负责向模型提供训练所需的数据。
- val_dataloader：此参数在执行器的 Runner.val() 方法中使用，并且会在 Runner.train() 方法的特定训练周期中被调用，用于对模型进行性能验证和评估。
- test_dataloader：此参数在执行器的 Runner.test() 方法中使用，用于对模型进行最终的测试。

MMEngine 兼容 PyTorch 的原生 DataLoader，这意味着你可以直接将已经创建的 DataLoader 实例传递给上述三个参数。

训练 Pipeline 配置通常包含以下几个步骤：

1. **加载图像**：定义从文件中读取图像的方式，包括解码后端和相关参数。
2. **加载标注**：定义读取图像对应的标注信息的方式，包括是否包含边界框等。
3. **数据增强**：定义各种数据增强操作，如随机HSV增强、Mosaic增强、随机尺寸调整、随机裁剪、随机翻转、填充和MixUp增强等。
4. **数据打包**：将处理后的图像和标注信息打包成模型输入格式。

以 RTMDet 为例，我们将在下面的示例中展示如何配置数据加载器。

### 加载图像

加载图像模块用于从文件中读取图像。

```python
dict(
    type=LoadImageFromFile,  # 加载图像的类型
    imdecode_backend="pillow",  # 解码后端
    backend_args=None,  # 后端参数
)
```

### 加载标注

加载标注模块用于读取图像对应的标注信息。

```python
dict(
    type=LoadAnnotations,  # 加载标注的类型
    imdecode_backend="pillow",  # 解码后端
    with_bbox=True,  # 是否包含边界框
)
```

### 数据增强

数据增强模块用于对图像进行各种增强操作，以提高模型的泛化能力。

```python
dict(type=HSVRandomAug),  # 随机HSV增强

dict(
    type=Mosaic,  # Mosaic增强
    img_scale=imgsz,  # 图像尺寸
    pad_val=114.0,  # 填充值
),

dict(
    type=RandomResize,  # 随机尺寸调整
    scale=(imgsz[0] * 2, imgsz[1] * 2),  # 尺寸范围
    ratio_range=(0.1, 2.0),  # 比例范围
    resize_type=Resize,  # 调整类型
    keep_ratio=True,  # 是否保持比例
),

dict(
    type=RandomCrop,  # 随机裁剪
    crop_size=imgsz,  # 裁剪尺寸
),

dict(
    type=RandomFlip,  # 随机翻转
    prob=0.5,  # 翻转概率
),

dict(
    type=Pad,  # 填充
    size=imgsz,  # 填充尺寸
    pad_val=dict(img=(114, 114, 114)),  # 填充值
),

dict(
    type=MixUp,  # MixUp增强
    img_scale=imgsz,  # 图像尺寸
    ratio_range=(1.0, 1.0),  # 比例范围
    max_cached_images=20,  # 最大缓存图像数量
    pad_val=114.0,  # 填充值
),
```

#### 数据打包

数据打包模块用于将处理后的图像和标注信息打包成模型输入格式。

```python
dict(type=PackDetInputs),  # 数据打包
```
