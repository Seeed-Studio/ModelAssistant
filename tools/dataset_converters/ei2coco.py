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

out_dir = args.out_directory
if os.path.exists(out_dir) and os.path.isdir(out_dir):
    shutil.rmtree(out_dir)

print('Transforming Edge Impulse data format into something compatible with Edgelab coco dataset format...')


def current_ms():
    return round(time.time() * 1000)

last_printed = current_ms()



def convert(path, category):
    global last_printed
    X = None
    class_count = 0
    converted_images = 0

    print('Converting ' + category + ' data...')
    
    with open(os.path.join(path, 'info.labels'), 'r') as f:
        X = json.loads(f.read())

    all_images = []
    classes = []
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
            "url": "https://seeedstduio.com"
        }],
        "type": "instances",
        "annotations": [],
        "categories": [],
    }

    images_dir = os.path.join(out_dir, category)
    os.makedirs(images_dir, exist_ok=True)

    total_images = len(X['files'])
    zf = len(str(total_images))

    for ix in range(0, total_images):
        img_file = X['files'][ix]['path']
        labels = X['files'][ix]['boundingBoxes']
        all_images.append(img_file)

        img_id = len(metadata['images']) + 1

        for lable in labels:
            
            if lable['label'] not in classes:
                classes.append(lable['label'])
                class_count = class_count + 1

            x = lable['x']
            y = lable['y']
            w = lable['width']
            h = lable['height']
            

            metadata['annotations'].append({
                "id": len(metadata['annotations']) + 1,
                "image_id": img_id,
                "category_id": classes.index(lable['label']) + 1,
                "bbox": [x, y, w, h],
                "area": w * h,
                "segmentation": [],
                "iscrowd": 0
            })
        
        im = Image.open(os.path.join(path, img_file))
        image_height = im.height
        image_width = im.width
        new_img_file = os.path.join(out_dir, category,  str(ix + 1).zfill(12) + '.jpg')
        im.save(new_img_file)
        im.close()
        
        metadata['images'].append({
            "id": img_id,
            "file_name": os.path.basename(new_img_file),
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
            "name": classes[c],
            "supercategory": ""
        })

    with open(annotations_file, 'w') as f:
        f.write(json.dumps(metadata, indent=4))


convert(os.path.join(args.data_directory, "training"), 'train')
convert(os.path.join(args.data_directory, "testing"), 'valid')

print('Transforming Edge Impulse data format into something compatible with Edgelab coco dataset format... done')
print('')
