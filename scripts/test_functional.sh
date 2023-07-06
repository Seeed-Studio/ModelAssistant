#!/bin/bash


# classification case
classification_test()
{
    # TODO
    exit 0
}

# detection case
detection_test()
{
    [ "$1" == "fetch" ] && \
        mkdir -p datasets && \
        wget https://files.seeedstudio.com/edgelab/datasets/COCO128.zip -v -O datasets/COCO128.zip && \
        unzip -fv datasets/COCO128.zip -d datasets/COCO128

    [ "$1" == "train" ] && \
        python3 tools/train.py \
            configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
            --no-validate \
            --cfg-options \
                data_root='datasets/COCO128' \
                epochs=1

    [ "$1" == "export" ] && \
        python3 tools/export.py \
            configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
            "$(cat work_dirs/yolov5_tiny_1xb16_300e_coco/last_checkpoint)" \
            tflite onnx \
            --calibration_epochs 1 \
            --cfg-options \
                data_root='datasets/COCO128'

    [ "$1" == "inference" ] && \
        python3 tools/inference.py \
            configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
            "$(cat work_dirs/yolov5_tiny_1xb16_300e_coco/last_checkpoint | sed -e 's/.pth/_int8.tflite/g')" \
            --cfg-options \
                data_root='datasets/COCO128' && \
        python3 tools/inference.py \
            configs/yolov5/yolov5_tiny_1xb16_300e_coco.py \
            "$(cat work_dirs/yolov5_tiny_1xb16_300e_coco/last_checkpoint | sed -e 's/.pth/_float32.onnx/g')" \
            --cfg-options \
                data_root='datasets/COCO128'
}


# pose case
pose_test()
{
    # TODO
    exit 0
}


# check args
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "$0: <TASK: [classification, detection, pose]> <STEP: [fetch, train, export, inference]>"
    exit 1
else
    case "$1" in
        "classification")
            classification_test "$2"
            ;;
        "detection")
            detection_test "$2"
            ;;
        "pose")
            pose_test "$2"
            ;;
    esac
fi
