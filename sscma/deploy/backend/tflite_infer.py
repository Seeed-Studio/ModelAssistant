# python test.py configs/rtmdet_nano_8xb32_300e_coco.py work_dirs/rtmdet_nano_8xb32_300e_coco/saved_model/epoch_300_float32.tflite --cfg-options test_evaluator.classwise=True

import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

from .base_infer import BaseInfer

import torch


class TFliteInfer(BaseInfer):
    def __init__(self, weights="sscma.tflite", device=torch.device("cpu")):
        super().__init__(weights=weights, device=device)
        self.interpreter = None
        self.input_details = None
        self.output_details = None

    def infer(self, input_data):
        results = []
        for data in input_data.split(1, 0):
            # check if input_data is Tensor
            if isinstance(data, torch.Tensor):
                # Torch Tensor convert NCWH to NHWC
                data = data.permute(0, 2, 3, 1).numpy()

                input = self.input_details[0]
                int8 = (
                    input["dtype"] == np.uint8 or input["dtype"] == np.int8
                )  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    data = (data / scale + zero_point).astype(
                        input["dtype"]
                    )  # de-scale
                self.interpreter.set_tensor(input["index"], data)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    # numpy x convert NHWC to NCWH
                    y.append(np.transpose(x, [0, 3, 1, 2]))

                if int8:
                    new_order = [4, 1, 5, 3, 0, 2]
                    new_y = [y[i] for i in new_order]
                    results.append(new_y)
                else:
                    results.append(y)

        return results

    def load_weights(self):
        self.interpreter = Interpreter(model_path=self.weights)  # load TFLite model

        self.interpreter.allocate_tensors()  # allocate
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs
