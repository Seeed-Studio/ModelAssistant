import numpy as np
import argparse
import math
import shutil
import os
import json
import time
from PIL import Image
from datetime import datetime

parser = argparse.ArgumentParser(
    description='Edge Impulse => Coco format converter')
parser.add_argument('--data-directory', type=str, required=True)
parser.add_argument('--out-directory', type=str, required=True)

args = parser.parse_args()

# Load data (images are in X_*.npy, labels are in JSON in Y_*.npy)
X_train = np.load(os.path.join(args.data_directory,
                  'X_split_train.npy'), mmap_mode='r')
X_test = np.load(os.path.join(args.data_directory,
                 'X_split_test.npy'), mmap_mode='r')

with open(os.path.join(args.data_directory, 'Y_split_train.npy'), 'r') as f:
    Y_train = json.loads(f.read())
with open(os.path.join(args.data_directory, 'Y_split_test.npy'), 'r') as f:
    Y_test = json.loads(f.read())

image_width, image_height, image_channels = list(X_train.shape[1:])

out_dir = args.out_directory
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

class_count = 0

print('Transforming Edge Impulse data format into something compatible with Edgelab coco dataset format...')


def current_ms():
    return round(time.time() * 1000)


total_images = len(X_train) + len(X_test)
zf = len(str(total_images))
last_printed = current_ms()
converted_images = 0


def convert(X, Y, category):
    global class_count, total_images, zf, last_printed, converted_images

    all_images = []
    annotations_file = os.path.join(out_dir, category, '_annotations.coco.json')
    if not os.path.exists(os.path.dirname(annotations_file)):
        os.makedirs(os.path.dirname(annotations_file), exist_ok=True)

    metadata = {
        "info": {
            "year": datetime.now().strftime("%Y"),
            "version": "1.0",
            "description": "Custom model",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "images": [],
        "licenses": [{
            "id": 1,
            "name": "Proprietary",
            "url": "https://edgeimpulse.com"
        }],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }

    for ix in range(0, len(X)):
        raw_img_data = (np.reshape(X[ix], (image_width, image_height, image_channels)) * 255).astype(np.uint8)
        labels = Y[ix]['boundingBoxes']

        images_dir = os.path.join(out_dir, category)
        os.makedirs(images_dir, exist_ok=True)

        img_file = os.path.join(images_dir, str(ix).zfill(12) + '.jpg')

        all_images.append(img_file)

        im = Image.fromarray(raw_img_data)
        im.save(img_file)

        img_id = len(metadata['images']) + 1

        for l in labels:
            if (l['label'] > class_count):
                class_count = l['label']

            x = l['x']
            y = l['y']
            w = l['w']
            h = l['h']

            metadata['annotations'].append({
                "id": len(metadata['annotations']) + 1,
                "image_id": img_id,
                "category_id": l['label'],
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": [],
                "iscrowd": 0
            })

        metadata['images'].append({
            "id": img_id,
            "license": 1,
            "file_name": os.path.basename(img_file),
            "height": image_height,
            "width": image_width,
            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        converted_images = converted_images + 1
        if (converted_images == 1 or current_ms() - last_printed > 3000):
            print('[' + str(converted_images).rjust(zf) + '/' + str(total_images) + '] Converting images...')
            last_printed = current_ms()

    for c in range(0, class_count):
        metadata['categories'].append({
            "id": c + 1,
            "name": str(c),
            "supercategory": str(c)
        })

    with open(annotations_file, 'w') as f:
        f.write(json.dumps(metadata, indent=4))


convert(X=X_train, Y=Y_train, category='train')
convert(X=X_test, Y=Y_test, category='valid')

print('[' + str(converted_images).rjust(zf) + '/' +
      str(total_images) + '] Converting images...')

print('Transforming Edge Impulse data format into something compatible with Edgelab coco dataset format... done')
print('')
