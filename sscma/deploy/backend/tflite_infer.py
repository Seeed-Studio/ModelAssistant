import numpy as np
import torch
import torchvision.transforms as transforms

from .base_infer import BaseInfer

from sscma.utils import lazy_import


class TFliteInfer(BaseInfer):

    @lazy_import("tflite_runtime", install_only=True)
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
                input = self.input_details[0]
                shape = input["shape"]
                if len(data.shape) == len(shape) and (data.shape[1] == 3 or data.shape[1] == 1):
                    w = shape[2]
                    h = shape[1]
                    # resize to model input shape
                    data = transforms.Resize((h, w))(data)
                # Torch Tensor convert NCWH to NHW
                data = data.permute(0, 2, 3, 1).numpy()

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

                results.append(y)
        return results

    def load_weights(self):
        from tflite_runtime.interpreter import Interpreter

        self.interpreter = Interpreter(model_path=self.weights)  # load TFLite model

        self.interpreter.allocate_tensors()  # allocate
        self.input_details = self.interpreter.get_input_details()  # inputs
        self.output_details = self.interpreter.get_output_details()  # outputs
