# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import Hook
from mmengine import HOOKS


@HOOKS.register_module()
class QuantizerSwitchHook(Hook):
    """Switch data pipeline at switch_epoch.

    Args:
        freeze_quantizer_epoch (int): the epoch to freeze the quantizer.
        freeze_bn_epoch (int): the epoch to freeze the batch normalization
            statistics.
    """

    def __init__(self,freeze_quantizer_epoch, freeze_bn_epoch):
        self.freeze_quantizer_epoch = freeze_quantizer_epoch
        self.freeze_bn_epoch = freeze_bn_epoch


    def before_train_epoch(self, runner):
        """switch pipeline."""
        epoch = runner.epoch
        #if epoch == runner.max_epoch // 3:
        if epoch == self.freeze_quantizer_epoch:
            runner.logger.info("freeze quantizer parameters")
            runner.model.apply(torch.quantization.disable_observer)
        #elif epoch == runner.max_epoch // 3 * 2:
        elif epoch == self.freeze_bn_epoch:
            runner.logger.info("freeze batch norm mean and variance estimates")
            runner.model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)