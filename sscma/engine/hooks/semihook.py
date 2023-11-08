from typing import Union
from mmengine.hooks import Hook
from mmengine.runner import Runner

from sscma.registry import HOOKS


@HOOKS.register_module()
class SemidHook(Hook):
    """ """

    def __init__(self, bure_epoch: Union[float, int] = 1) -> None:
        super().__init__()
        if isinstance(bure_epoch, float):
            assert (
                bure_epoch <= 1.0
            ), "The number of supervised training rounds must be less than the maximum number of rounds"

        self.bure_epoch = bure_epoch

    def before_run(self, runner: Runner) -> None:
        if isinstance(self.bure_epoch, float):
            self.bure_epoch = int(runner.max_epochs * self.bure_epoch)

        assert self.bure_epoch <= runner.max_epochs

    def before_train_epoch(self, runner: Runner) -> None:
        if self.bure_epoch == runner.epoch:
            # dataloader starts loading unlabeled dataset for semi-supervised training
            runner.train_dataloader.sampler.with_unlabel = True
