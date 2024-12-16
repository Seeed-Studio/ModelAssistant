from typing import Union, List, Dict, Tuple
import torch
import numpy as np

from mmengine.registry import MODELS
from mmengine.model import BaseModel
from sscma.utils.typing_utils import OptConfigType, OptMultiConfig
from sscma.structures import DetDataSample, OptSampleList
from sscma.models.heads.pfld_head import pose_acc
from sscma.structures import PoseDataSample
from mmengine.structures.instance_data import InstanceData

ForwardResults = Union[
    Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor
]


class PFLDQuantModel(BaseModel):
    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        tinynn_model: torch.nn.Module = None,
        loss_cfg: dict = dict(type="PFLDLoss"),
    ):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self._model = tinynn_model
        self.lossFunction = MODELS.build(loss_cfg)
        # self.bbox_head = bbox_head

    def forward(
        self, inputs: torch.Tensor, data_samples: OptSampleList, mode: str = "predict"
    ):
        if mode == "predict":
            return self.predict(inputs, data_samples)
        elif mode == "loss":
            return self.loss(inputs, data_samples)

    def loss(self, inputs: torch.Tensor, data_samples: OptSampleList):
        preds = self._model(inputs)
        labels = torch.as_tensor(
            data_samples["keypoints"], device=preds.device, dtype=torch.float32
        )
        loss = self.lossFunction(preds, labels)
        acc = pose_acc(preds, labels, data_samples["hw"])
        return {"loss": loss, "Acc": torch.as_tensor(acc, dtype=torch.float32)}

    def predict(self, inputs: torch.Tensor, data_samples: OptSampleList):
        data = self._model(inputs)
        res = PoseDataSample(**data_samples)
        res.results = data
        res.pred_instances = InstanceData(
            keypoints=np.array([data.reshape(-1, 2).cpu().numpy()])
            * data_samples["init_size"][1].reshape(-1, 1).cpu().numpy()
        )

        return [res]

    def set_model(self, model):
        self._model = model
