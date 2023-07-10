import random

import torch
import torch.nn.functional as F

from .helper_funcs import AugBasic
from .resample import Resampler


def pad_sample_seq_batch(x, n_samples):
    if x.size(0) >= n_samples:
        max_x_start = x.size(0) - n_samples
        x_start = random.randint(0, max_x_start)
        x = x[:, x_start : x_start + n_samples]
    else:
        x = F.pad(x, (0, n_samples - x.size(1)), "constant").data
    return x


def batch_resample(Resample, data, seq_len):
    data = Resample(data.squeeze(1))
    data = pad_sample_seq_batch(data, seq_len)
    data = data.unsqueeze(1)
    return data


class BatchAugs(AugBasic):
    def __init__(self, params):
        super().__init__(fs=params['fs'])
        self.params = params
        self.params['fft_params'] = self.fft_params
        self.params['fft_params'] = self.fft_params

        if self.params['mix_loss'] == 'ce':
            self.loss = torch.nn.NLLLoss(reduction='none')
        elif self.params['mix_loss'] == 'bce':
            self.loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
        else:
            raise ValueError("Wrong mix_loss")

        if len(params['resample_factors']) > 0:
            self.random_resample = [
                Resampler(input_sr=params['fs'], output_sr=int(params['fs'] * fac), dtype=torch.float32).to(
                    params['device']
                )
                for fac in params['resample_factors']
            ]
        else:
            self.random_resample = []

    def __call__(self, x, y, epoch):
        """resample."""
        if len(self.random_resample) > 0 and random.random() < 0.5:
            R = self.random_resample[random.randint(0, len(self.random_resample) - 1)]
            x = batch_resample(R, x, self.params['seq_len'])
        '''mix'''
        if (
            len(self.params['augs']) > 0
            and random.random() <= self.params['mix_ratio']
            and epoch > self.params['epoch_mix']
        ):
            is_mixed = True
            i = random.randint(0, len(self.params['augs']) - 1)
            aug = self.params['augs'][i]
            if aug == 'mixup':
                x, y = self.mixup(x, y)
            elif aug == 'timemix':
                x, y = self.timemix(x, y)
            elif aug == 'phmix':
                x, y = self.phmix(x, y)
            elif aug == 'freqmix':
                x, y = self.freqmix(x, y)
            else:
                raise ValueError("wrong mix aug")
        else:
            is_mixed = False
        return x, y, is_mixed

    def mixup(self, data, target):
        idx = torch.randperm(data.size(0))
        data_shuffled = data[idx, ...].clone()
        target_shuffled = target[idx].clone()
        lam = 0.1 + 0.9 * torch.rand(data.shape[0], 1, 1, device=data.device, requires_grad=False)
        G = 10 * torch.log10(torch.clamp((data**2).mean(-1, keepdims=True), min=1e-5))
        G_shuffled = G[idx]
        p = 1 / (1 + 10 ** ((G - G_shuffled) / 20) * (1 - lam) / lam)
        data = (data * p + data_shuffled * (1 - p)) / torch.sqrt(p**2 + (1 - p) ** 2)
        targets = (target, target_shuffled, p.view(-1))
        targets = [t.to(data.device) for t in targets]
        return data, targets

    def timemix(self, data, target):
        idx = torch.randperm(data.size(0))
        data_shuffled = data[idx].clone()
        target_shuffled = target[idx].clone()
        a = 0.5
        lam = a * torch.rand(data.shape[0], 1, 1, device=data.device, requires_grad=False) + (1 - a)
        n = data.shape[-1]
        n1 = (n * (1 - lam)).view(-1).int()
        for k, nn in enumerate(n1):
            if random.random() < 0.5:
                data[k, :, n - nn :] = data_shuffled[k, :, n - nn :].clone()
            else:
                data[k, :, :nn] = data_shuffled[k, :, :nn].clone()
        del data_shuffled
        targets = (target, target_shuffled, lam.view(-1))
        targets = [t.to(data.device) for t in targets]
        return data, targets

    def freqmix(self, data, target):
        data = data.squeeze(1)
        idx = torch.randperm(data.size(0))
        idx_win = random.randint(0, len(self.params['fft_params']['win_len']) - 1)
        win = torch.hann_window(self.params['fft_params']['win_len'][idx_win]).to(data.device)
        X = torch.stft(
            data,
            win_length=self.params['fft_params']['win_len'][idx_win],
            hop_length=self.params['fft_params']['hop_len'][idx_win],
            n_fft=self.params['fft_params']['n_fft'][idx_win],
            return_complex=True,
            window=win,
        )
        X_shuffled = X[idx, ...].clone()
        target_shuffled = target[idx].clone()
        a = 0.5
        lam = a * torch.rand(X.shape[0], 1, 1, device=X.device, requires_grad=False) + (1 - a)
        n = X.shape[1]
        n1 = (n * (1 - lam)).view(-1).int()
        for k in range(X.shape[0]):
            nn = n1[k]
            if random.random() < 0.5:
                X[k, n - nn :, :] = X_shuffled[k, n - nn :, :].clone()
            else:
                X[k, :nn, :] = X_shuffled[k, :nn, :].clone()
        del X_shuffled
        data = torch.istft(
            X,
            win_length=self.params['fft_params']['win_len'][idx_win],
            hop_length=self.params['fft_params']['hop_len'][idx_win],
            n_fft=self.params['fft_params']['n_fft'][idx_win],
            window=win,
        )
        targets = (target, target_shuffled, lam.view(-1))
        data = data.unsqueeze(1)
        targets = [t.to(data.device) for t in targets]
        return data, targets

    def phmix(self, data, target):
        data = data.squeeze(1)
        b, device = data.shape[0], data.device
        idx = torch.randperm(data.size(0))
        idx_win = random.randint(0, len(self.params['fft_params']['win_len']) - 1)
        target_shuffled = target[idx].clone()
        X = torch.stft(
            data,
            win_length=self.params['fft_params']['win_len'][idx_win],
            hop_length=self.params['fft_params']['hop_len'][idx_win],
            n_fft=self.params['fft_params']['n_fft'][idx_win],
            return_complex=True,
        )
        X_ph = X.angle()
        X_shuffled_ph = X_ph[idx, ...].clone()
        lam = torch.rand(b, 1, 1, device=device, requires_grad=False)
        ph = X_ph.mul_(lam).add_((1 - lam) * X_shuffled_ph)
        X = X.abs() * torch.exp(1j * ph)
        data = torch.istft(
            X,
            win_length=self.params['fft_params']['win_len'][idx_win],
            hop_length=self.params['fft_params']['hop_len'][idx_win],
            n_fft=self.params['fft_params']['n_fft'][idx_win],
        )
        targets = (target, target_shuffled, lam.view(-1) * 0.5 + 0.5)
        data = data.unsqueeze(1)
        targets = [t.to(data.device) for t in targets]
        return data, targets

    def mix_loss(self, logits, target, n_classes=None, pred_one_hot=None):
        target, target_shuffled, lam = target
        lam = lam.view(-1, 1)
        if self.params['mix_loss'] == 'ce':
            log_p = F.log_softmax(logits, dim=-1)
            loss = self.loss(log_p, target)
            loss_mix = self.loss(log_p, target_shuffled)
            loss = (loss * lam + loss_mix * (1 - lam)).mean(0)
        elif self.params['mix_loss'] == 'bce':
            if not pred_one_hot:
                target = F.one_hot(target, n_classes).float()
                target_shuffled = F.one_hot(target_shuffled, n_classes).float() * (lam < 0.9)
                one_h_mix = torch.clamp(target + target_shuffled, max=1)
                loss = self.loss(logits, one_h_mix)
            else:
                target_shuffled *= lam < 0.9
                one_h_mix = torch.clamp(target + target_shuffled, max=1)
                loss = self.loss(logits, one_h_mix)
        return loss


if __name__ == '__main__':
    pass
