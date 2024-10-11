import os.path as osp

import tensorflow as tf
import torch

from .base_infer import BaseInfer


class SavedModelInfer(BaseInfer):
    def __init__(self, weights="sscma.pb", device=torch.device("cpu")):
        super().__init__(weights=weights, device=device)
        self.model = None

    def infer(self, input_data):
        results = []
        for data in input_data.split(1, 0):
            # check if input_data is Tensor
            if isinstance(data, torch.Tensor):
                # Torch Tensor convert NCWH to NHWC
                data = data.permute(0, 2, 3, 1).numpy()
            inputs = {"inputs": tf.convert_to_tensor(data)}
            ordered_result = self.model(**inputs)
            # convert NHWC to NCWH
            ordered_result = [tf.transpose(v, [0, 3, 1, 2]) for v in ordered_result]
            # convert tf tensor to numpy
            results.append([item.numpy() for item in ordered_result])

        return results

    def load_weights(self):
        self.model = tf.saved_model.load(osp.dirname(self.weights))
