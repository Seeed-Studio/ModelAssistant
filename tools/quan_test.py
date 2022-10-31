import os
from typing import *
import ncnn
import numpy as np

import onnx
import tqdm.std
from PIL import Image
import onnxruntime
from torchvision.transforms import *
import tensorflow as tf

input_name = 'images'
output_name = 'output'


def read_img(p):
    img = Image.open(p)
    trans = Compose([ToTensor(), Resize((112, 112)), Grayscale()])
    img = trans(img).cpu().numpy()
    # inimg = ncnn.Mat.from_pixels_resize(img,ncnn.Mat.PixelType.PIXEL_RGB,img.shape[1],img.shape[2],224,224)
    mat_img = ncnn.Mat(img)
    return img, mat_img


class Inter():
    def __init__(self, model: List or AnyStr):
        if isinstance(model, list):
            net = ncnn.Net()
            for p in model:
                if p.endswith('param'): param = p
                if p.endswith('bin'): bin = p
            net.load_param(param)
            net.load_model(bin)
        elif model.endswith('onnx'):
            try:
                net = onnx.load(model)
                onnx.checker.check_model(net)
            except:
                raise 'onnx file have error,please check your onnx export code!'
            net = onnxruntime.InferenceSession(model)
        elif model.endswith('tflite'):
            inter = tf.lite.Interpreter
            net = inter(model)
            net.allocate_tensors()
        else:
            raise 'model file input error'
        self.inter = net

    def __call__(self, img: np.array, input_name: AnyStr = 'input', output_name: AnyStr = 'output'):
        if len(img.shape) == 2:  # audio
            if img.shape[0] > 10:
                img = img.transpose(1, 0)
        else:  # image
            C, H, W = img.shape
            if C not in [1, 3]:
                img = img.transpose(2, 0, 1)
            img = np.array([img])

        if isinstance(self.inter, onnxruntime.InferenceSession):  # onnx
            result = self.inter.run([output_name], {input_name: img})[0][0]
        elif isinstance(self.inter, ncnn.Net):  # ncnn
            self.inter.opt.use_vulkan_compute = False
            extra = self.inter.create_extractor()
            extra.input(input_name, ncnn.Mat(img))
            result = extra.extract(output_name)[1]
            result = [result[i]for i in range(len(result))]
        else:  # tf
            input_, output = self.inter.get_input_details()[0], self.inter.get_output_details()[0]
            int8 = input_['dtype'] == np.int8 or input_['dtype'] == np.uint8
            if int8:
                scale, zero_point = input_['quantization']
                img = (img.transpose(0, 2, 3, 1) / scale + zero_point).astype(np.int8)
            self.inter.set_tensor(input_['index'], img)
            self.inter.invoke()
            result = self.inter.get_tensor(output['index'])
            if int8:
                scale, zero_point = output['quantization']
                result = (result.astype(np.float32) - zero_point) * scale

        return result


if __name__ == '__main__':
    flls = os.listdir('./img_e')
    flls = [os.path.join('./img_e', i) for i in flls if i.endswith('.jpg')]
    ncnn_quan = []
    ncnn_float = []
    onnx_qu = []
    onnx_float = []
    tf_quan = []
    ncnn_model = Inter(['./weights/best.param', './weights/best.bin'])
    ncnn_model_quan = Inter(['./weights/best-int8.param', './weights/best-int8.bin'])
    onnx_model = Inter('./weights/best.onnx')
    onnx_model_quan = Inter('./weights/best_int8.onnx')
    tf_model = Inter('./weights/best-int8.tflite')

    for i in tqdm.std.tqdm(flls):
        img, float_inimg = read_img(i)
        r1 = onnx_model(img, input_name)
        r2 = onnx_model_quan(img, input_name)
        n1 = ncnn_model(img, input_name)
        n2 = ncnn_model_quan(img, input_name)
        t = tf_model(img, input_name)
        ncnn_quan.append(n2)
        ncnn_float.append(n1)
        onnx_qu.append(r2)
        onnx_float.append(r1)
        tf_quan.append(t)

    onnx_float = np.array(onnx_float)
    onnx_qu = np.array(onnx_qu)
    ncnn_float = np.array(ncnn_float)
    ncnn_quan = np.array(ncnn_quan)
    tf_quan = np.array(tf_quan)

    print('onnx:', np.sum(np.square(onnx_float - onnx_qu)))
    print('ncnn', np.sum(np.square(ncnn_float - ncnn_quan)))
    print("onnx-tf", np.sum(np.square(onnx_float - tf_quan)))
    print('onnx-ncnn', np.sum(np.square(onnx_float - ncnn_quan)))
    print('ncnn-onnx', np.sum(np.square(ncnn_float - onnx_qu)))
    print('ncnn-onnx', np.sum(np.square(ncnn_float - onnx_float)))
