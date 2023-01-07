#!/bin/bash
set -e

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
OUT_DIRECTORY=$(realpath 'out')

cd /app/Edgelab

export PYTHONPATH=/app/Edgelab:$PYTHONPATH

echo "Training..."

python3 tools/train.py mmdet configs/fomo/fomo_mobnetv2_x8_coco.py --cfg-options runner.max_epochs=1 data.train.dataset.pipeline.4.img_scale=\(192,192\) data.val.pipeline.1.img_scale=\(192,192\) data.test.pipeline.1.img_scale=\(192,192\) data.train.dataset.data_root=/scripts/dataset data.val.data_root=/scripts/dataset data.test.data_root=/scipts/dataset

echo "Training complete"
echo ""

mkdir -p $OUT_DIRECTORY
cp -r /app/Edgelab/work_dirs/ $OUT_DIRECTORY