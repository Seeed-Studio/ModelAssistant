# python test.py configs/rtmdet_nano_8xb32_300e_coco.py work_dirs/rtmdet_nano_8xb32_300e_coco/saved_model --cfg-options test_evaluator.classwise=True

from collections import OrderedDict
import tensorflow as tf

from .base_infer import BaseInfer

import torch


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
            result = self.model(tf.convert_to_tensor(data))
            # sort key output_0 to output_6
            ordered_result = OrderedDict(
                sorted(result.items(), key=lambda x: int(x[0].split("_")[1]))
            )
            # convert NHWC to NCWH
            ordered_result = {
                k: tf.transpose(v, [0, 3, 1, 2]) for k, v in ordered_result.items()
            }
            # convert tf tensor to numpy
            results.append([item.numpy() for item in ordered_result.values()])
            # results.append([)

        return results

    def load_weights(self):
        self.model = tf.saved_model.load(self.weights).signatures["serving_default"]
