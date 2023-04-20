import torch
from mmcls.models.builder import build_loss
from ..utils.batch_augs import BatchAugs
from mmengine.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.runner import Runner


@HOOKS.register_module()
class Audio_hooks(Hook):
    """
    Considering that the corresponding data enhancement 
    will be done according to the current epoch during the 
    training process, it is necessary to modify the corresponding 
    hook in the hook
    """

    def __init__(self, n_cls, multilabel, loss, seq_len, sampling_rate, device,
                 augs_mix, mix_ratio, local_rank, epoch_mix, mix_loss):
        super(Audio_hooks, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        ba_params = {
            'seq_len': seq_len,
            'fs': sampling_rate,
            'device': self.device,
            'augs': augs_mix,
            'mix_ratio': mix_ratio,
            'batch_sz': local_rank,
            'epoch_mix': epoch_mix,
            'resample_factors': [0.8, 0.9, 1.1, 1.2],
            'multilabel': True if multilabel else False,
            'mix_loss': mix_loss
        }
        self.batch_augs = BatchAugs(ba_params)
        self.n_cls = n_cls
        self.mutilabel = multilabel
        self.loss = build_loss(loss)

    def before_train_iter(self, runner: Runner, batch_idx, data_batch):
        (x, y) = data_batch.values()
        (x_k, y_k) = data_batch.keys()
        epoch = runner.epoch
        x, self.targets, self.is_mixed = self.batch_augs(
            x.to(self.device), y.to(self.device), epoch)

        runner._train_loop.data_batch = {x_k: x, y_k: y.float()}
        # runner.targets = targets
        # setattr(runner,'targets',targets)

    def after_train_iter(self, runner: Runner, batch_idx, data_batch, outputs):

        if self.is_mixed:
            loss_cls = self.batch_augs.mix_loss(outputs['inputs'],
                                                self.targets,
                                                self.n_cls,
                                                pred_one_hot=self.mutilabel)
        else:
            loss_cls = self.loss(**outputs)
        acc = (outputs['targets'] == torch.max(outputs['inputs'],
                                               dim=1)[1]).float().mean()

        # runner.message_hub.update_scalars({'loss': loss_cls, 'acc': acc})
        runner.message_hub.update_info_dict({'loss': loss_cls, 'acc': acc})
