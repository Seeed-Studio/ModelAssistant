# YOLOv3

> [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)

<!-- [ALGORITHM] -->

## Abstract

We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate. It's still fast though, don't worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5 mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster.

<div align=center>
<img src="https://user-images.githubusercontent.com/40661020/144001433-b4f7fb5e-3b7a-414b-b949-93733213b670.png" height="300"/>
</div>

## Results and Models

|  Backbone   | Scale | Lr schd | Mem (GB) | Inf time (fps) | box AP |                                                          Config                                                          |                                                                                                                                                                        Download                                                                                                                                                                        |
| :---------: | :---: | :-----: | :------: | :------------: | :----: | :----------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| MobileNetV2 |  416  |  300e   |   5.3    |                |  24.1  | [config](./yolov3_mbv2_416_coco.py) | [model](https://github.com/Seeed-Studio/edgelab/releases/download/model_zoo/yolov3_mbv2_416_coco.pth)|

Notice: We reduce the number of channels to 96 in both head and neck. It can reduce the flops and parameters, which makes these models more suitable for edge devices.

## Credit

This implementation originates from the project of Haoyu Wu(@wuhy08) at Western Digital.

## Citation

```latex
@misc{redmon2018yolov3,
    title={YOLOv3: An Incremental Improvement},
    author={Joseph Redmon and Ali Farhadi},
    year={2018},
    eprint={1804.02767},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
