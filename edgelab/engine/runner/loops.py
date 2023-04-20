from typing import Sequence

from mmengine.runner.loops import EpochBasedTrainLoop

from edgelab.registry import LOOPS


@LOOPS.register_module()
class GetEpochBasedTrainLoop(EpochBasedTrainLoop):


    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.data_batch = data_batch
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=self.data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        self.outputs = self.runner.model.train_step(
            self.data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=self.data_batch,
            outputs=self.outputs)
        self._iter += 1
