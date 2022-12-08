import torch
from mmcls.models.builder import build_loss
from core.utils.batch_augs import BatchAugs
from mmcv.runner import Hook, HOOKS, EpochBasedRunner


@HOOKS.register_module()
class Audio_hooks(Hook):
    def __init__(self, n_cls, multilabel, loss, seq_len, sampling_rate, device, augs_mix, mix_ratio, local_rank,
                 epoch_mix, mix_loss):
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

    def before_train_iter(self, runner: EpochBasedRunner):
        (x, y) = runner.data_batch.values()
        (x_k, y_k) = runner.data_batch.keys()
        epoch = runner.epoch
        x, targets, is_mixed = self.batch_augs(
            x.to(self.device), y.to(self.device), epoch)
            
        runner.data_batch = {x_k: x, y_k: y.float()}
        runner.targets = targets
        runner.is_mixed = is_mixed

    def after_train_iter(self, runner: EpochBasedRunner):
        outputs = runner.outputs
        if runner.is_mixed:
            loss_cls = self.batch_augs.mix_loss(
                outputs['inputs'], runner.targets, self.n_cls, pred_one_hot=self.mutilabel)
        else:
            loss_cls = self.loss(**outputs)
        acc = (outputs['targets'] == torch.max(
            outputs['inputs'], dim=1)[1]).float().mean()
        runner.outputs = {'loss': loss_cls, 'acc': acc}
