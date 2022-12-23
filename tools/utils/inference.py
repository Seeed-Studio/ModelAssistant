import os.path as osp
from typing import List, AnyStr, Tuple

import cv2
import mmcv
import onnx
import torch
import numpy as np

from models.utils.computer_acc import pose_acc


class Inter():

    def __init__(self, model: List or AnyStr or Tuple):
        if isinstance(model, list):
            try:
                import ncnn
            except:
                raise ImportError(
                    'You have not installed ncnn yet, please execute the "pip install ncnn" command to install and run again'
                )
            net = ncnn.Net()
            for p in model:
                if p.endswith('param'): param = p
                if p.endswith('bin'): bin = p
            net.load_param(param)
            net.load_model(bin)
            # net.opt.use_vulkan_compute = True
            self.engine = 'ncnn'
        elif model.endswith('onnx'):
            try:
                import onnxruntime
            except:
                raise ImportError(
                    'You have not installed onnxruntime yet, please execute the "pip install onnxruntime" command to install and run again'
                )
            try:
                net = onnx.load(model)
                onnx.checker.check_model(net)
            except:
                raise ValueError(
                    'onnx file have error,please check your onnx export code!')
            providers = [
                'CUDAExecutionProvider', 'CPUExecutionProvider'
            ] if torch.cuda.is_available() else ['CPUExecutionProvider']
            net = onnxruntime.InferenceSession(model, providers=providers)
            self.engine = 'onnx'
        elif model.endswith('tflite'):
            try:
                import tensorflow as tf
            except:
                raise ImportError(
                    'You have not installed tensorflow yet, please execute the "pip install tensorflow" command to install and run again'
                )
            inter = tf.lite.Interpreter
            net = inter(model)
            net.allocate_tensors()
            self.engine = 'tf'
        else:
            raise 'model file input error'
        self.inter = net

    def __call__(self,
                 img: np.array,
                 input_name: AnyStr = 'input',
                 output_name: AnyStr = 'output'):
        if len(img.shape) == 2:  # audio
            if img.shape[0] > 10:
                img = img.transpose(1, 0)
        else:  # image
            C, H, W = img.shape
            if C not in [1, 3]:
                print(img.shape)
                img = img.transpose(2, 1, 0)
            img = np.array([img])

        if self.engine == 'onnx':  # onnx
            result = self.inter.run(
                [self.inter.get_outputs()[0].name],
                {self.inter.get_inputs()[0].name: img})[0][0]
        elif self.engine == 'ncnn':  # ncnn
            self.inter.opt.use_vulkan_compute = False
            extra = self.inter.create_extractor()
            extra.input(input_name, ncnn.Mat(img[0]))
            result = extra.extract(output_name)[1]
            result = [result[i] for i in range(len(result))]
        else:  # tf
            input_, output = self.inter.get_input_details(
            )[0], self.inter.get_output_details()[0]
            int8 = input_['dtype'] == np.int8 or input_['dtype'] == np.uint8
            img = img.transpose(0, 2, 3, 1)
            if int8:
                scale, zero_point = input_['quantization']
                img = (img / scale + zero_point).astype(np.int8)
            self.inter.set_tensor(input_['index'], img)
            self.inter.invoke()
            result = self.inter.get_tensor(output['index'])
            if int8:
                scale, zero_point = output['quantization']
                result = (result.astype(np.float32) - zero_point) * scale

        return result


def inference_test(model, data_loader):
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        # parse data
        input = data['img']
        target = data['keypoints'].cpu().numpy()
        size = data['hw']  #.cpu().numpy()
        input = input.cpu().numpy()
        result = model(input[0])
        result = np.array(result)
        if len(result.shape) == 1:
            result = np.array([result])
        acc = pose_acc(result.copy(), target, size)
        results.append({
            'Acc': acc,
            'pred': result,
            'image_file': data['image_file'].data[0][0]
        })

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    print(results)
    return results


def show_point(keypoints,
               img_file,
               win_name='test',
               save_path=False,
               not_show=False):
    pass
    img = mmcv.imread(img_file, channel_order='bgr').copy()
    h, w = img.shape[:-1]
    keypoints[:, ::2] = keypoints[:, ::2] * w
    keypoints[:, 1::2] = keypoints[:, 1::2] * h

    for point in keypoints:
        if not isinstance(point, (float, int)):
            img = cv2.circle(img, (int(point[0]), int(point[1])), 2,
                             (255, 0, 0), -1)
    if not not_show:
        cv2.imshow(win_name, img)
        cv2.waitKey(500)

    if save_path:
        img_name = osp.basename(img_file)
        cv2.imwrite(osp.join(save_path, img_name), img)
