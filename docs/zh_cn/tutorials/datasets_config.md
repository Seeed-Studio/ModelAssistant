# 如何使用自定义数据集

如果用户在没有自定义数据集加载器的情况下，同时想使用已有的模型训练自己的数据集，这需要用户将自己的数据集按照一定的格式存储，并修改模型配置文件的几个参数方可完成。以下会告诉用户如何定义自己的数据以及如何修改配置文件并开始训练自己的数据集。本项目的自定义数据集的使用遵循mmlab的要求，用户也可参照[mmlab相关教程](https://mmdetection.readthedocs.io/zh_CN/latest/)使用。

## 数据集支持的格式

针对目标检测、分类、和关键点检测任务，每种任务的数据集格式定义都不相同，在此我们分别描述每种数据集的格式。

### 1.目标检测

目前支持的数据格式有COCO和PASCAL，

1. 假设用户使用图像数据集，可通过将数据集上传值[roboflow](https://app.roboflow.com/)。同时可做适当修改等操作。此时下载数据可选择coco格式。
2. 用户也可写相关转换脚本，将自己的数据集转换值COCO格式，COCO格式的数据集的必要字段如下所示：

```json
'images': [
    {
        'file_name': 'COCO_val2014_000000001268.jpg',
        'height': 427,
        'width': 640,
        'id': 1268
    },
    ...
],

'annotations': [
    {
        'segmentation': [[192.81,
            247.09,
            ...
            219.03,
            249.06]],  # 如果有 mask 标签
        'area': 1035.749,
        'iscrowd': 0,
        'image_id': 1268,
        'bbox': [192.81, 224.8, 74.73, 33.43],
        'category_id': 16,
        'id': 42986
    },
    ...
],

'categories': [
    {'id': 0, 'name': 'car'},
 ]
```

在 json 文件中有三个必要的键：

- `images`: 包含多个图片以及它们的信息的数组，其中 `file_name`、`height`、`width` 和 `id`字段是必要的。
- `annotations`: 包含多个实例标注信息的数组,如果是目标检测任务，则必须包含`bbox`、`image_id`、`category_id`、`id`字段。
- `categories`: 包含多个类别名字和 ID 的数组。

### 2.分类

### 3.关键点检测

## 修改模型配置文件

### 1.目标检测

对于配置文件的修改主要修改模型类别和数据集位置指定的位置。
本示例使用yolov3模型的配置文件修改为例，对于需要修改的位置如下所示：

```python
_base_ = '../_base_/pose_default_runtime.py'
custom_imports = dict(imports=['models', 'datasets'],
                      allow_failed_imports=False)
# 模型配置，下面需要修改模型的头的分类数量，需要和你使用的数据类别的数量相等
model = dict(
    type='YOLOV3',
    backbone=dict(type='MobileNetV2',
                  out_indices=(2, 4, 6),
                  act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                  init_cfg=dict(type='Pretrained',
                                checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(type='YOLOV3Neck',
              num_scales=3,
              in_channels=[320, 96, 32],
              out_channels=[96, 96, 96]),
    # 这里是模型头的定义，需要将num_classes的值修改为自己数据的类别数量，对于有多个头的模型，则需要修改每个头的类被数量。
    bbox_head=dict(type='YOLOV3Head',
                   num_classes=80, <<------ 例如你的数据集类别为10，需要将这里的80修改为10即可，模型部分的即可修改完毕
                   in_channels=[96, 96, 96],
                   out_channels=[96, 96, 96],
                   anchor_generator=dict(type='YOLOAnchorGenerator',
                                         base_sizes=[[(116, 90), (156, 198),
                                                      (373, 326)],
                                                     [(30, 61), (62, 45),
                                                      (59, 119)],
                                                     [(10, 13), (16, 30),
                                                      (33, 23)]],
                                         strides=[32, 16, 8]),
                   bbox_coder=dict(type='YOLOBBoxCoder'),
                   featmap_strides=[32, 16, 8],
                   loss_cls=dict(type='CrossEntropyLoss',
                                 use_sigmoid=True,
                                 loss_weight=1,
                                 reduction='sum'),
                   loss_conf=dict(type='CrossEntropyLoss',
                                  use_sigmoid=True,
                                  loss_weight=1.0,
                                  reduction='sum'),
                   loss_xy=dict(type='CrossEntropyLoss',
                                use_sigmoid=True,
                                loss_weight=2.0,
                                reduction='sum'),
                   loss_wh=dict(type='MSELoss',
                                loss_weight=2.0,
                                reduction='sum')),
    # training and testing settings
    train_cfg=dict(assigner=dict(
        type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0)),
    test_cfg=dict(nms_pre=1000,
                  min_bbox_size=0,
                  score_thr=0.05,
                  conf_thr=0.005,
                  nms=dict(type='nms', iou_threshold=0.45),
                  max_per_img=100))
# dataset settings
dataset_type = 'CustomCocoDataset'

# 需要修改数据集的根目录，但不是必要的，下面会解释
data_root = '["http://images.cocodataset.org/zips/train2017.zip", "http://images.cocodataset.org/zips/val2017.zip", "http://images.cocodataset.org/zips/test2017.zip", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"]'

# 这里是数据集的配置处，这里必须修改的有ann_file和img_prefix两个值，如果这两个值的路径是绝对路径，则可以不指定data_root的值
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',  # use RepeatDataset to speed up training
        times=10,
        dataset=dict(type=dataset_type,
                     classes=('cat','dog' ...),    <<------ 这里需要修改为你数据集的类别，格式为数组,也可在上方指定，这里只需使用对应变量替换即可
                     data_root=data_root,      <<------ 这里的修改可以不是必要的，如果没有指定值，那么下面两个值得路径则必须为绝对路径
                     ann_file='annotations/instances_train2017.json', <<------ 这里需要修改为你训练数据集的注释文件路径
                     img_prefix='train2017/',             <<------ 这里需要修改为你训练数据集图片路径
                     pipeline=train_pipeline)),
    val=dict(type=dataset_type,
             classes=('cat','dog' ...),     <<------ 这里需要修改为你数据集的类别，格式为数组,也可在上方指定，这里只需使用对应变量替换即可
             data_root=data_root,       <<------ 这里的修改可以不是必要的，如果没有指定值，那么下面两个值得路径则必须为绝对路径
             ann_file='annotations/instances_val2017.json',     <<------ 这里需要修改为你的验证数据集的注释文件路径
             img_prefix='val2017/',     <<------ 这这里需要修改为你的验证数据集图片路径
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              classes=('cat','dog' ...),    <<------ 这里需要修改为你数据集的类别，格式为数组,也可在上方指定，这里只需使用对应变量替换即可
              data_root=data_root,      <<------ 这里的修改可以不是必要的，如果没有指定值，那么下面两个值得路径则必须为绝对路径
              ann_file='annotations/instances_val2017.json',<<------ 这里需要修改为你的验证数据集的注释文件路径
              img_prefix='val2017/',    <<------ 这里需要修改为你的验证训练数据集图片路径
              pipeline=test_pipeline))

```

## 开始训练

在数据集格式转换完成和配置文件修改完成后，即可开始训练，训练命令如下所示：

```shell
python tools/train.py $TASK $CONFIG_PATH --workdir=$WORKERDIR --gpus=1 #使用cpu可设置为0
```

### 参数解释

- `$TASK`：训练的任务类型，可在`mmdet`、`mmcls`、`mmpose`中选择一个，分别代表目标检测、分类、关键点检测。
- `$CONFIG_PATH`：上面修改后的配置文件路径。
