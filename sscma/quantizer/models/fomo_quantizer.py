from typing import Union, List, Dict, Tuple
import torch
from torch.nn import functional as F

from mmengine.registry import MODELS
from mmengine.model import BaseModel
from sscma.utils.typing_utils import OptConfigType
from sscma.structures import DetDataSample, OptSampleList
from sscma.utils.misc import samplelist_boxtype2tensor, unpack_gt_instances

ForwardResults = Union[
    Dict[str, torch.Tensor], List[DetDataSample], Tuple[torch.Tensor], torch.Tensor
]


class FomoQuantizer(BaseModel):
    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        tinynn_model: torch.nn.Module = None,
        head: torch.nn.Module = None,
        skip_preprocessor: bool = True,
    ):
        super().__init__(data_preprocessor=data_preprocessor)
        self._model = tinynn_model
        self.head = MODELS.build(head)
        self.skip_preprocessor = skip_preprocessor
        # self.bbox_head = bbox_head

    def forward(
        self, inputs: torch.Tensor, data_samples: OptSampleList, mode: str = "predict"
    ):
        if isinstance(inputs, list):
            inputs = torch.stack(inputs, dim=0).to(self.data_preprocessor.device)
       
        if inputs.dtype == torch.uint8:
            inputs = inputs / 255
        elif inputs.dtype == torch.int8:
            inputs = inputs / 128

        if mode == "loss":
            pred = self._model(inputs)
            gt = unpack_gt_instances(data_samples)
            (batch_gt_instances, batch_gt_instances_ignore, batch_img_metas) = gt
            loss = self.head.loss_by_feat(
                [pred],
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore,
            )
            return loss
        elif mode == "predict":
            preds = self._model(inputs).unsqueeze(0)
            results_list = self.head.predict_by_feat(preds, data_samples)
            for data_sample, pred_instances in zip(data_samples, results_list):
                data_sample.pred_instances = pred_instances
            samplelist_boxtype2tensor(data_samples)

            return data_samples

    def _loss(self, inputs: torch.Tensor, batch_data_samples: OptSampleList):
        data = self._model(inputs)
        # Fast version
        loss_inputs = data + (
            batch_data_samples["bboxes_labels"],
            batch_data_samples["img_metas"],
        )
        losses = self.bbox_head.loss_by_feat(*loss_inputs)
        return losses

    def loss(self, inputs: torch.Tensor, batch_data_samples: OptSampleList):
        data = self._model(inputs)
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        results = self.bbox_head.predict_by_feat(data, batch_img_metas)
        results = samplelist_boxtype2tensor(results, batch_data_samples)
        return results

    def set_model(self, model):
        self._model = model

    def train_step(
        self, data: Union[dict, tuple, list], optim_wrapper
    ) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            if not self.skip_preprocessor:
                data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode="loss")  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        if not self.skip_preprocessor:
            data = self.data_preprocessor(data, False)
        return self._run_forward(data, "predict")

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        if not self.skip_preprocessor:
            data = self.data_preprocessor(data, False)
        return self._run_forward(data, "predict")
