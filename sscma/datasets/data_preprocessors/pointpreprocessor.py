# Copyright (c) Seeed Tech Ltd. All rights reserved.
from typing import Optional, Union

import torch
from mmengine.logging import MessageHub
from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor

from sscma.engine.utils.batch_augs import BatchAugs
from sscma.registry import MODELS


@MODELS.register_module()
class ETADataPreprocessor(BaseDataPreprocessor):
    def __init__(
        self,
        n_cls,
        multilabel,
        seq_len,
        sampling_rate,
        augs_mix,
        mix_ratio,
        local_rank,
        epoch_mix,
        mix_loss,
        non_blocking: Optional[bool] = False,
    ):
        self.n_cls = n_cls
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ba_params = {
            'seq_len': seq_len,
            'fs': sampling_rate,
            'device': self._device,
            'augs': augs_mix,
            'mix_ratio': mix_ratio,
            'batch_sz': local_rank,
            'epoch_mix': epoch_mix,
            'resample_factors': [0.8, 0.9, 1.1, 1.2],
            'multilabel': True if multilabel else False,
            'mix_loss': mix_loss,
        }
        super().__init__(non_blocking)

        self.audio_augs = BatchAugs(ba_params)
        self.messbus = MessageHub.get_current_instance()
        self.messbus.update_info('audio_loss', self.audio_augs)

    def forward(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = super().cast_data(data)
        (x, y) = data.values()
        (x_k, y_k) = data.keys()

        epoch = MessageHub.get_current_instance().get_info('epoch')

        x, target, ismixed = self.audio_augs(x.to(self.device), y.to(self.device), epoch)

        self.messbus.update_info('target', target)
        self.messbus.update_info('ismixed', ismixed)
        return {x_k: x, y_k: y.float()}
