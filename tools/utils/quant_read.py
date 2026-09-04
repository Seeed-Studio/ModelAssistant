# Copyright (c) Seeed Technology Co.,Ltd. All rights reserved.
import os
import random

import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')


def find_and_sample_images(folder_path, limit=10000, sample_size=100):
    """Recursively find images in ``folder_path`` and randomly sample them.

    Backported from main (dea366b, issue #294): a flat ``os.listdir`` with a
    narrow extension filter silently finds zero images when the calibration
    folder contains subdirectories or other image formats, which then
    produces a broken quantized model without any error.
    """
    image_files = []

    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided image path '{folder_path}' is not a valid directory.")

    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMG_EXTENSIONS:
                image_files.append(os.path.join(root, file))
                if len(image_files) >= limit:
                    break
        if len(image_files) >= limit:
            break

    found = len(image_files)
    if found < sample_size:
        if found == 0:
            raise ValueError(f"No images found in the directory '{folder_path}'.")
        print(f"Warning: Found only {found} images, which is less than the requested sample size of {sample_size}.")
        sample_size = found

    return random.sample(image_files, sample_size)


class Quan_Reader(CalibrationDataReader):
    def __init__(self, images_folder, size, input_name, batch_size=1) -> None:
        # super(CalibrationDataReader).__init__(self)
        self.images_folder = images_folder
        self.size = size
        self.input_name = input_name
        self.transfor = Compose([ToTensor(), Grayscale(), Resize(size=size)])
        self.num = 0

        self.enum_data_dicts = None
        self.init()

    def init(self):
        self.file_ls = iter(find_and_sample_images(self.images_folder, limit=10000, sample_size=10000))

    def get_next(self) -> dict:
        try:
            a = next(self.file_ls)
            if a is None:
                raise StopIteration
            img = self.process_data(a)
            return {self.input_name: np.array([img])}
        except Exception:
            return None

    def process_data(self, file):
        img = Image.open(file)
        img = self.transfor(img).cpu().numpy()
        return img
