import numpy as np
import torch
import torchaudio
import random
import scipy
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from edgelab.registry import TRANSFORMS


class AugBasic:
    def __init__(self, fs):
        super().__init__()
        self.fs = fs
        self.fft_params = {}
        if fs == 22050:
            self.fft_params['win_len'] = [512, 1024, 2048]
            self.fft_params['hop_len'] = [128, 256, 1024]
            self.fft_params['n_fft'] = [512, 1024, 2048]
        elif fs == 16000:
            self.fft_params['win_len'] = [256, 512, 1024]
            self.fft_params['hop_len'] = [256 // 4, 512 // 4, 1024 // 4]
            self.fft_params['n_fft'] = [256, 512, 1024]
        elif fs == 8000:
            self.fft_params['win_len'] = [128, 256, 512]
            self.fft_params['hop_len'] = [32, 64, 128]
            self.fft_params['n_fft'] = [128, 256, 512]
        else:
            raise ValueError


class RandomRIR(AugBasic):
    def __init__(self, fs, p=0.5):
        self.p = p
        self.fs = fs

    def rir(self, mic, n, r, rm, src):
        nn = np.arange(-n, n + 1, 1).astype(np.float32)
        srcs = np.power(-1, nn)
        rms = nn + 0.5 - 0.5 * srcs
        xi = srcs * src[0] + rms * rm[0] - mic[0]
        yj = srcs * src[1] + rms * rm[1] - mic[1]
        zk = srcs * src[2] + rms * rm[2] - mic[2]
        [i, j, k] = np.meshgrid(xi, yj, zk)
        d = np.sqrt(i ** 2 + j ** 2 + k ** 2)
        t = np.round(self.fs * d / 343.) + 1
        [e, f, g] = np.meshgrid(nn, nn, nn)
        c = np.power(r, np.abs(e) + np.abs(f) + np.abs(g))
        e = c / d
        y = np.ones_like(d).reshape(-1).astype(np.int32)
        t = t.reshape(-1).astype(np.int32)
        e = e.reshape(-1)
        h = coo_matrix((e, (t, y))).todense()[:, 1]
        h = np.array(h).ravel()
        h = h / np.abs(h).max()
        if h.shape[0] % 2 == 0:
            h = h[:-1]
        return h

    def __call__(self, sample):
        if random.random() < self.p:
            r = 2 * np.random.rand(1) - 1
            n = 3

            x = 20 * np.random.rand(1)
            y = 20 * np.random.rand(1)
            z = 4 * np.random.rand(1)
            rm = np.array([x, y, z])

            x = rm[0] * np.random.rand(1)
            y = rm[1] * np.random.rand(1)
            z = rm[2] * np.random.rand(1)

            mic = np.array([x, y, z])
            x = rm[0] * np.random.rand(1)
            y = rm[1] * np.random.rand(1)
            z = rm[2] * np.random.rand(1)

            src = np.array([x, y, z])

            h = self.rir(mic, n, r, rm, src)
            h = torch.from_numpy(h).float()
            sample = sample[None, None, :]
            sample = F.pad(sample, (h.shape[-1] // 2, h.shape[-1] // 2), "reflect")
            sample = F.conv1d(sample, h[None, None, :], bias=None, stride=1, padding=0, dilation=1,
                              groups=sample.shape[1])
        return sample, h


class RandomLPHPFilter(AugBasic):
    def __init__(self, fs, p=0.5, fc_lp=None, fc_hp=None):
        self.p = p
        self.fs = fs
        self.fc_lp = fc_lp
        self.fc_hp = fc_hp
        self.num_taps = 15

    def __call__(self, sample):
        if random.random() < self.p:
            a = 0.25
            if random.random() < 0.5:
                fc = 0.5 + random.random() * 0.25
                filt = scipy.signal.firwin(self.num_taps, fc, window='hamming')
            else:
                fc = random.random() * 0.25
                filt = scipy.signal.firwin(self.num_taps, fc, window='hamming', pass_zero=False)
            filt = torch.from_numpy(filt).float()
            filt = filt / filt.sum()
            sample = F.pad(sample.view(1, 1, -1), (filt.shape[0] // 2, filt.shape[0] // 2), mode="reflect")
            sample = F.conv1d(sample, filt.view(1, 1, -1), stride=1, groups=1)
            sample = sample.view(-1)
        return sample


class RandomTimeShift(AugBasic):
    def __init__(self, p=0.5, max_time_shift=None):
        self.p = p
        self.max_time_shift = max_time_shift

    def __call__(self, sample):
        if random.random() < self.p:
            if self.max_time_shift is None:
                self.max_time_shift = sample.shape[-1] // 10
            int_d = 2 * random.randint(0, self.max_time_shift) - self.max_time_shift
            frac_d = np.round(100 * (random.random() - 0.5)) / 100
            if int_d + frac_d == 0:
                return sample
            if int_d > 0:
                pad = torch.zeros(int_d, dtype=sample.dtype)
                sample = torch.cat((pad, sample[:-int_d]), dim=-1)
            elif int_d < 0:
                pad = torch.zeros(-int_d, dtype=sample.dtype)
                sample = torch.cat((sample[-int_d:], pad), dim=-1)
            else:
                pass
            if frac_d == 0:
                return sample
            n = sample.shape[-1]
            dw = 2 * np.pi / n
            if n % 2 == 1:
                wp = torch.arange(0, np.pi, dw)
                wn = torch.arange(-dw, -np.pi, -dw).flip(dims=(-1,))
            else:
                wp = torch.arange(0, np.pi, dw)
                wn = torch.arange(-dw, -np.pi - dw, -dw).flip(dims=(-1,))
            w = torch.cat((wp, wn), dim=-1)
            phi = frac_d * w
            sample = torch.fft.ifft(torch.fft.fft(sample) * torch.exp(-1j * phi)).real
        return sample


class RandomTimeMasking(AugBasic):
    def __init__(self, p=0.5, n_mask=None):
        self.n_mask = n_mask
        self.p = p

    def __call__(self, sample):
        if self.n_mask is None:
            self.n_mask = int(0.05 * sample.shape[-1])
        if random.random() < self.p:
            max_start = sample.size(-1) - self.n_mask
            idx_rand = random.randint(0, max_start)
            sample[idx_rand:idx_rand + self.n_mask] = torch.randn(self.n_mask) * 1e-6
        return sample


class RandomMuLawCompression(AugBasic):
    def __init__(self, p=0.5, n_channels=256):
        self.n_channels = n_channels
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            e = torchaudio.functional.mu_law_encoding(sample, self.n_channels)
            sample = torchaudio.functional.mu_law_decoding(e, self.n_channels)
        return sample


class RandomAmp(AugBasic):
    def __init__(self, low, high, p=0.5):
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            amp = torch.FloatTensor(1).uniform_(self.low, self.high)
            sample.mul_(amp)
        return sample


class RandomFlip(AugBasic):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.data = torch.flip(sample.data, dims=[-1, ])
        return sample


class RandomAdd180Phase(AugBasic):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample.mul_(-1)
        return sample


class RandomAdditiveWhiteGN(AugBasic):
    def __init__(self, p=0.5, snr_db=30):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            w = torch.randn_like(sample).mul_(sgm)
            sample.add_(w)
        return sample


class RandomAdditiveUN(AugBasic):
    def __init__(self, snr_db=35, p=0.5):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.) * np.sqrt(3)
            w = torch.rand_like(sample).mul_(2 * sgm).add_(-sgm)
            sample.add_(w)
        return sample


class RandomAdditivePinkGN(AugBasic):
    def __init__(self, snr_db=35, p=0.5):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] / k.sqrt()
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAdditiveVioletGN(AugBasic):
    def __init__(self, p=0.5, snr_db=35):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] * k
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAdditiveRedGN(AugBasic):
    def __init__(self, p=0.5, snr_db=35):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] / k
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomAdditiveBlueGN(AugBasic):
    def __init__(self, p=0.5, snr_db=35):
        self.snr_db = snr_db
        self.min_snr_db = 30
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            s = torch.sqrt(torch.mean(sample ** 2))
            n = sample.shape[-1]
            w = torch.randn(n)
            nn = n // 2 + 1
            k = torch.arange(1, nn + 1, 1).float()
            W = torch.fft.fft(w)
            W = W[:nn] * k.sqrt()
            W = torch.cat((W, W.flip(dims=(-1,))[1:-1].conj()), dim=-1)
            w = torch.fft.ifft(W).real
            w.add_(w.mean()).div_(w.std())
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            sgm = s * 10 ** (-snr_db / 20.)
            sample.add_(w.mul_(sgm))
        return sample


class RandomFreqShift(AugBasic):
    def __init__(self, sgm, fs, p=0.5):
        super().__init__(fs=fs)
        self.sgm = sgm
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            win_idx = random.randint(0, len(self.fft_params['win_len']) - 1)
            df = self.fs / self.fft_params['win_len'][win_idx]
            f_shift = torch.randn(1).mul_(self.sgm * df)
            t = torch.arange(0, self.fft_params['win_len'][win_idx], 1).float()
            w = torch.real(torch.exp(-1j * 2 * np.pi * t * f_shift))
            X = torch.stft(sample,
                           win_length=self.fft_params['win_len'][win_idx],
                           hop_length=self.fft_params['hop_len'][win_idx],
                           n_fft=self.fft_params['n_fft'][win_idx],
                           window=w,
                           return_complex=True)
            sample = torch.istft(X,
                                 win_length=self.fft_params['win_len'][win_idx],
                                 hop_length=self.fft_params['hop_len'][win_idx],
                                 n_fft=self.fft_params['n_fft'][win_idx])

        return sample


class RandomAddSine(AugBasic):
    def __init__(self, fs, snr_db=35, max_freq=50, p=0.5):
        self.snr_db = snr_db
        self.max_freq = max_freq
        self.min_snr_db = 30
        self.p = p
        self.fs = fs

    def __call__(self, sample):
        n = torch.arange(0, sample.shape[-1], 1)
        f = self.max_freq * torch.rand(1) + 3 * torch.randn(1)
        if random.random() < self.p:
            snr_db = self.min_snr_db + torch.rand(1) * (self.snr_db - self.min_snr_db)
            t = n * 1. / self.fs
            s = (sample ** 2).mean().sqrt()
            sgm = s * np.sqrt(2) * 10 ** (-snr_db / 20.)
            b = sgm * torch.sin(2 * np.pi * f * t + torch.rand(1) * np.pi)
            sample.add_(b)

        return sample


class RandomAmpSegment(AugBasic):
    def __init__(self, low, high, max_len=None, p=0.5):
        self.low = low
        self.high = high
        self.max_len = max_len
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            if self.max_len is None:
                self.max_len = sample.shape[-1] // 10
            idx = random.randint(0, self.max_len)
            amp = torch.FloatTensor(1).uniform_(self.low, self.high)
            sample[idx: idx + self.max_len].mul_(amp)
        return sample


class RandomPhNoise(AugBasic):
    def __init__(self, fs, sgm=0.01, p=0.5):
        super().__init__(fs=fs)
        self.sgm = sgm
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            win_idx = random.randint(0, len(self.fft_params['win_len']) - 1)
            sgm_noise = self.sgm + 0.01 * torch.rand(1)
            X = torch.stft(sample,
                           win_length=self.fft_params['win_len'][win_idx],
                           hop_length=self.fft_params['hop_len'][win_idx],
                           n_fft=self.fft_params['n_fft'][win_idx],
                           return_complex=True)
            w = sgm_noise * torch.rand_like(X)
            phn = torch.exp(1j * w)
            X.mul_(phn)
            sample = torch.istft(X,
                                 win_length=self.fft_params['win_len'][win_idx],
                                 hop_length=self.fft_params['hop_len'][win_idx],
                                 n_fft=self.fft_params['n_fft'][win_idx])
        return sample


class RandomCyclicShift(AugBasic):
    def __init__(self, max_time_shift=None, p=0.5):
        self.max_time_shift = max_time_shift
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            if self.max_time_shift is None:
                self.max_time_shift = sample.shape[-1]
            int_d = random.randint(0, self.max_time_shift - 1)
            if int_d > 0:
                sample = torch.cat((sample[-int_d:], sample[:-int_d]), dim=-1)
            else:
                pass
        return sample


@TRANSFORMS.register_module()
class AudioAugs():
    def __init__(self, k_augs):
        self.noise_vec = ['awgn', 'abgn', 'apgn', 'argn', 'avgn', 'aun', 'phn', 'sine']
        self.k_augs = k_augs

    def _init(self, fs, p=0.5, snr_db=30):
        augs = {}
        for aug in self.k_augs:
            if aug == 'amp':
                augs['amp'] = RandomAmp(p=p, low=0.5, high=1.3)
            elif aug == 'flip':
                augs['flip'] = RandomFlip(p)
            elif aug == 'neg':
                augs['neg'] = RandomAdd180Phase(p)
            elif aug == 'awgn':
                augs['awgn'] = RandomAdditiveWhiteGN(p=p, snr_db=snr_db)
            elif aug == 'abgn':
                augs['abgn'] = RandomAdditiveBlueGN(p=p, snr_db=snr_db)
            elif aug == 'argn':
                augs['argn'] = RandomAdditiveRedGN(p=p, snr_db=snr_db)
            elif aug == 'avgn':
                augs['avgn'] = RandomAdditiveVioletGN(p=p, snr_db=snr_db)
            elif aug == 'apgn':
                augs['apgn'] = RandomAdditivePinkGN(p=p, snr_db=snr_db)
            elif aug == 'mulaw':
                augs['mulaw'] = RandomMuLawCompression(p=p, n_channels=256)
            elif aug == 'tmask':
                augs['tmask'] = RandomTimeMasking(p=p, n_mask=int(0.1 * fs))
            elif aug == 'tshift':
                augs['tshift'] = RandomTimeShift(p=p, max_time_shift=int(0.1 * fs))
            elif aug == 'sine':
                augs['sine'] = RandomAddSine(p=p, fs=fs)
            elif aug == 'cycshift':
                augs['cycshift'] = RandomCyclicShift(p=p, max_time_shift=None)
            elif aug == 'ampsegment':
                augs['ampsegment'] = RandomAmpSegment(p=p, low=0.5, high=1.3, max_len=int(0.1 * fs))
            elif aug == 'aun':
                augs['aun'] = RandomAdditiveUN(p=p, snr_db=snr_db)
            elif aug == 'phn':
                augs['phn'] = RandomPhNoise(p=p, fs=fs, sgm=0.01)
            elif aug == 'fshift':
                augs['fshift'] = RandomFreqShift(fs=fs, sgm=1, p=p)
            else:
                raise ValueError("{} not supported".format(aug))
        self.augs = augs
        self.augs_signal = [a for a in augs if a not in self.noise_vec]
        self.augs_noise = [a for a in augs if a in self.noise_vec]

    def __call__(self, sample, fs, p=0.5, snr_db=30, **kwargs):
        self._init(fs=fs, p=p, snr_db=snr_db)
        augs = self.augs_signal
        augs_noise = self.augs_noise
        random.shuffle(augs)
        if len(augs_noise) > 0:
            i = random.randint(0, len(augs_noise) - 1)
            augs.append(augs_noise[i])
        for aug in augs:
            sample = self.augs[aug](sample)
        return sample


if __name__ == "__main__":
    r = RandomRIR(fs=22050, p=1)
    x = torch.zeros(22050)
    x[0:100] = 1
    y = r(x)
    import matplotlib.pyplot as plt

    plt.plot(x)
    plt.plot(y[0].view(-1), 'r')
    plt.show()


