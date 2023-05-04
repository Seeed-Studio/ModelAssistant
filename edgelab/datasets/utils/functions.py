import torch
from mmengine.registry import FUNCTIONS
from mmdet.structures import DetDataSample

@FUNCTIONS.register_module()
def fomo_collate(batch):
    img, label = [x['inputs'] for x in batch], [y['data_samples'] for y in batch]
    # for i, l in enumerate(label):
    #     if l.shape[0] > 0:
    #         l[:, 0] = i
    return dict(inputs=torch.stack(img), data_samples=[DetDataSample(labels=torch.cat(label, 0))])
 