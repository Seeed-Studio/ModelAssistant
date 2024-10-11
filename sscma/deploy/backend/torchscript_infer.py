import torch
from .base_infer import BaseInfer


class TorchScriptInfer(BaseInfer):
    def __init__(self, weights="sscma.torchscript", device=torch.device("cpu")):
        super().__init__(weights=weights, device=device)
        self.model = None

    def infer(self, input_data):
        results = []
        for data in input_data.split(1, 0):
            results.append(self.model(data))
        return results

    def load_weights(self):
        self.model = torch.jit.load(self.weights)
