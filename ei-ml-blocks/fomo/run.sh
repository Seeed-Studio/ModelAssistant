#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd $SCRIPTPATH

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case $1 in
    --epochs) # e.g. 50
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
    --learning-rate) # e.g. 0.01
      LEARNING_RATE="$2"
      shift # past argument
      shift # past value
      ;;
    --data-directory) # e.g. /tmp/data
      DATA_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    --out-directory) # e.g. out
      OUT_DIRECTORY="$2"
      shift # past argument
      shift # past value
      ;;
    *)
      POSITIONAL_ARGS+=("$1") # save positional arg
      shift # past argument
      ;;
  esac
done

if [ -z "$EPOCHS" ]; then
    echo "Missing --epochs"
    exit 1
fi
if [ -z "$LEARNING_RATE" ]; then
    echo "Missing --learning-rate"
    exit 1
fi
if [ -z "$DATA_DIRECTORY" ]; then
    echo "Missing --data-directory"
    exit 1
fi
if [ -z "$OUT_DIRECTORY" ]; then
    echo "Missing --out-directory"
    exit 1
fi

OUT_DIRECTORY=$(realpath $OUT_DIRECTORY)
DATA_DIRECTORY=$(realpath $DATA_DIRECTORY)

# convert Edge Impulse dataset (in Numpy format, with JSON for labels into coco dataset format)
python3 -u extract_dataset.py --data-directory $DATA_DIRECTORY --out-directory /tmp/data

# Disable W&B prompts
export WANDB_MODE=disabled

NUM_CLASS=`jq '.categories | length' /tmp/data/train/_annotations.coco.json`
IMG_HIGHT=`jq '.images[0].height' /tmp/data/train/_annotations.coco.json`
IMG_WIDTH=`jq '.images[0].width' /tmp/data/train/_annotations.coco.json`

cd /app/Edgelab

# add the current directory to the PYTHONPATH
export PYTHONPATH=/app/Edgelab:$PYTHONPATH

python3 -u tools/train.py mmdet configs/fomo/fomo_mobnetv2_x8_coco.py \
    --cfg-options \
    runner.max_epochs=$EPOCHS   \
    optimizer.lr=$LEARNING_RATE \
    data.train.dataset.pipeline.4.img_scale=\($IMG_HIGHT,$IMG_WIDTH\) \
    data.val.pipeline.1.img_scale=\($IMG_HIGHT,$IMG_WIDTH\) \
    data.test.pipeline.1.img_scale=\($IMG_HIGHT,$IMG_WIDTH\) \
    data.train.dataset.data_root=/tmp/data \
    data.val.data_root=/tmp/data \
    data.test.data_root=/tmp/data 


echo "Training complete"
echo ""

mkdir -p $OUT_DIRECTORY

#copy the model to the output directory
cp /app/Edgelab/work_dirs/fomo_mobnetv2_x8_coco/exp1/latest.pth $OUT_DIRECTORY/model.pth

# convert to .onnx
# python3 tools/torch2onnx.py --task mmdet --shape $IMG_HIGHT --config configs/fomo/fomo_mobnetv2_x8_coco.py --checkpoint work_dirs/fomo_mobnetv2_x8_coco/exp1/latest.pth 
# cp /app/Edgelab/work_dirs/fomo_mobnetv2_x8_coco/exp1/latest.onnx $OUT_DIRECTORY/model.onnx

python3 ./tools/export.py mmdet /app/Edgelab/work_dirs/fomo_mobnetv2_x8_coco/exp1/fomo_mobnetv2_x8_coco.py  --tflite_type fp32 --weights $OUT_DIRECTORY/model.pth --shape $IMG_HIGHT --data /tmp/data/train

mv $OUT_DIRECTORY/model_fp32.tflite $OUT_DIRECTORY/model.tflite

python3 ./tools/export.py mmdet /app/Edgelab/work_dirs/fomo_mobnetv2_x8_coco/exp1/fomo_mobnetv2_x8_coco.py  --tflite_type int8 --weights $OUT_DIRECTORY/model.pth --shape $IMG_HIGHT --data /tmp/data/train

mv $OUT_DIRECTORY/model_int8.tflite $OUT_DIRECTORY/model_quantized_int8_io.tflite
