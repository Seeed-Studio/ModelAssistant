import torch
import numpy as np

from .base_infer import BaseInfer

class HailoInfer(BaseInfer):

    def __init__(self, weights="sscma.har", device="cpu", script=None):
        super().__init__()
        # self.runner = Client#Runner(har=weights)
        self.device = device
        if script is not None:
            self.runner.load_model_script(script)

    def infer(self, input_data):
        results = []
        B, _, _, _ = input_data.shape
        # if isinstance(input_data, torch.Tensor):
        #     input_data = input_data.detach().cpu().numpy()
        # with self.runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
        #     input_data = input_data.transpose(0, 2, 3, 1)
        #     res = self.runner.infer(ctx, input_data)
        #     for result in zip(*[np.split(r, B) for r in res]):
        #         results.append([s.transpose(0, 3, 1, 2) for s in result])

        return results

    def load_weights(self):
        return super().load_weights()
