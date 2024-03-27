import copy
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

epsilon = 1e-8


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


def count_parameters(model):
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_weights_for_balanced_classes(samples, nclasses):
    count = [0] * nclasses
    for item in samples:
        count[item[1]] += 1
    weight_per_class = [0.0] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(samples)
    for idx, val in enumerate(samples):
        weight[idx] = weight_per_class[val[1]]
    return weight


def measure_inference_time(model, input, repetitions=300, use_16b=False):
    device = torch.device('cuda')
    model_ = copy.deepcopy(model)
    model_.eval()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    # repetitions = 300
    timings = np.zeros((repetitions, 1))

    if use_16b:
        input = input.half()
        model_.half()
    else:
        pass
    input = input.to(device)
    model_.to(device)
    for _ in range(10):
        _ = model_(input)
    with torch.no_grad():
        # GPU-WARM-UP
        for rep in range(repetitions):
            starter.record()
            _ = model_(input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    return mean_syn, std_syn


def collate_fn(batch):
    x = [item[0] for item in batch]
    y = [item[1] for item in batch]
    x = torch.stack(x, dim=0).contiguous()
    return (x, y)


def files_to_list(filename):
    """Takes a text file of filenames and makes a list of filenames."""
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files


def find_first_nnz(t, q, dim=1):
    _, mask_max_indices = torch.max(t == q, dim=dim)
    return mask_max_indices


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    with torch.no_grad():
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].view(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def average_precision(output, target):
    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))
    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)
    return precision_at_i


def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """
    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean()


def pad_sample_seq(x, n_samples):
    if x.size(-1) >= n_samples:
        max_x_start = x.size(-1) - n_samples
        x_start = random.randint(0, max_x_start)
        x = x[x_start : x_start + n_samples]
    else:
        x = F.pad(x, (0, n_samples - x.size(-1)), 'constant').data
    return x


def pad_sample_seq_batch(x, n_samples):
    if x.size(0) >= n_samples:
        max_x_start = x.size(0) - n_samples
        x_start = random.randint(0, max_x_start)
        x = x[:, x_start : x_start + n_samples]
    else:
        x = F.pad(x, (0, n_samples - x.size(1)), 'constant').data
    return x


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        # print(name)
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0}, {'params': decay, 'weight_decay': weight_decay}]


def _get_bn_param_ids(net):
    bn_ids = []
    for m in net.modules():
        print(m)
        if isinstance(m, torch.nn.BatchNorm1d) or isinstance(m, torch.nn.LayerNorm):
            bn_ids.append(id(m.weight))
            bn_ids.append(id(m.bias))
        elif isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                bn_ids.append(id(m.bias))
    return bn_ids


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def gather_tensor(tensor, n):
    rt = tensor.clone()
    tensor_list = [torch.zeros(n, device=tensor.device, dtype=torch.cuda.float()) for _ in range(n)]
    dist.all_gather(tensor_list, rt)
    return tensor_list


def parse_gpu_ids(gpu_ids):  # list of ints
    s = ''.join(str(x) + ',' for x in gpu_ids)
    s = s.rstrip().rstrip(',')
    return s


def representative_dataset(dataset):
    # load data from dataset in cfg.
    for i, fn in enumerate(dataset):
        if 'img' in fn.keys():
            data = fn['img']
            if not isinstance(data, torch.Tensor):  # for yolov3
                data = data[0].data
            data = data.permute(1, 2, 0)
        else:
            data = fn['audio']
            data = data.permute(1, 0)

        data = data.cpu().numpy()
        data = np.expand_dims(data, axis=0)
        yield [data]
        if i >= 100:
            break
