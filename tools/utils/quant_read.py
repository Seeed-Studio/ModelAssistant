# Copyright (c) Seeed Tech Ltd. All rights reserved.
import os

import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor

img_format = ['.JPG', '.PNG', '.JPEG']


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
        file_ls = os.listdir(self.images_folder)
        self.file_ls = iter(
            [os.path.join(self.images_folder, i) for i in file_ls if os.path.splitext(i)[-1].upper() in img_format]
        )

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
