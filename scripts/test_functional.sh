#!/bin/bash


# classification case
classification_test()
{
    # TODO
    echo -e "Warning: classification test not implemented"
    CONFIG_FILE=""
    DATASETS_URL=""

    # functional_test_core "$1" "${CONFIG_FILE}" "${DATASETS_URL}"
    return $?
}


# detection case
detection_test()
{
    CONFIG_FILE="configs/yolov5/yolov5_tiny_1xb16_300e_coco.py"
    DATASETS_URL="https://files.seeedstudio.com/edgelab/datasets/COCO128.zip"

    functional_test_core "$1" "${CONFIG_FILE}" "${DATASETS_URL}"
    return $?
}


# pose case
pose_test()
{
    # TODO
    echo -e "Warning: pose test not implemented"
    CONFIG_FILE=""
    DATASETS_URL=""

    # functional_test_core "$1" "${CONFIG_FILE}" "${DATASETS_URL}"
    return $?
}


# functional test core
# args: $0 STEP CONFIG_FILE DATASETS_URL
functional_test_core()
{
    CONFIG_FILE="$2"
    DATASETS_URL="$3"

    CONFIG_FILE_NAME="$(basename -- ${CONFIG_FILE})"
    DATASETS_PATH="datasets/$(basename -- ${DATASETS_URL})"
    DATASETS_DIR="${DATASETS_PATH%.*}"
    LAST_CHECKPOINT="work_dirs/${CONFIG_FILE_NAME%.*}/last_checkpoint"

    echo -e "CONFIG_FILE=${CONFIG_FILE}"
    echo -e "DATASETS_DIR=${DATASETS_DIR}"
    echo -e "LAST_CHECKPOINT=${LAST_CHECKPOINT}"

    case "$1" in
        "fetch")
            mkdir -p datasets && \
            wget "${DATASETS_URL}" -v -O "${DATASETS_PATH}" && \
            unzip -o "${DATASETS_PATH}" -d "${DATASETS_DIR}"
            return $?
            ;;
        "train")
            python3 tools/train.py \
                "${CONFIG_FILE}" \
                --no-validate \
                --cfg-options \
                    data_root="${DATASETS_DIR}" \
                    max_epochs=10
            return $?
            ;;
        "export")
            tree work_dirs
            python3 tools/export.py \
                "${CONFIG_FILE}" \
                "$(cat \"${LAST_CHECKPOINT} \")" \
                tflite onnx \
                --calibration_epochs 1 \
                --cfg-options \
                    data_root="${DATASETS_DIR}"
            return $?
            ;;
        "inference")
            tree work_dirs
            python3 tools/inference.py \
                "${CONFIG_FILE}" \
                "$(cat ${LAST_CHECKPOINT} | sed -e 's/.pth/_int8.tflite/g')" \
                --cfg-options \
                    data_root="${DATASETS_DIR}" \
            && \
            python3 tools/inference.py \
                "${CONFIG_FILE}" \
                "$(cat ${LAST_CHECKPOINT} | sed -e 's/.pth/_float32.onnx/g')" \
                --cfg-options \
                    data_root="${DATASETS_DIR}"
            return $?
            ;;
        *)
            echo -e "Supported steps: ['fetch', 'train', 'export', 'inference'], expected '$1'"
            return 1
            ;;
    esac
}


# check args
if [ -z "$1" ] || [ -z "$2" ]; then
    echo -e "Usage: bash $0 \"<TASK>\" \"<STEP>\""
    echo -e "\t TASK: ['classification', 'detection', 'pose']"
    echo -e "\t STEP: ['fetch', 'train', 'export', 'inference']"
    exit 1
else
    case "$1" in
        "classification")
            classification_test "$2"
            exit $?
            ;;
        "detection")
            detection_test "$2"
            exit $?
            ;;
        "pose")
            pose_test "$2"
            exit $?
            ;;
        *)
            echo -e "Supported tasks: ['classification', 'detection', 'pose'], expected '$2'"
            exit 1
            ;;
    esac
fi
