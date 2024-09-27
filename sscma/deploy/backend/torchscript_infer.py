from statsmodels.graphics.tukeyplot import results

from .base_infer import BaseInfer

import torch


class TorchScriptInfer(BaseInfer):
    def __init__(self, weights="sscma.torchscript", device=torch.device("cpu")):
        super().__init__(weights=weights, device=device)
        self.model = None

    def infer(self, input_data):
        # input_data: NCHW
        # separate NCHW to 1CHW to infer very image
        # cls_scores = []
        # bbox_preds = []
        # for data in input_data.split(1, 0):
        #     results = self.model(data)
        #     cls_scores.append(results[0])
        #     bbox_preds.append(results[1])
        #
        # return cls_scores, bbox_preds
        results = []
        for data in input_data.split(1, 0):
            results.append(self.model(data))
        return results

    def load_weights(self):
        self.model = torch.jit.load(self.weights)
