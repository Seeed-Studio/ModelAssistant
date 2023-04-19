import torch
from mmengine.registry import FUNCTIONS

@FUNCTIONS.register_module()
def fomo_collate(batch):
    img, label = [x['img'] for x in batch], [y['target'] for y in batch]
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return dict(img=torch.stack(img), target=torch.cat(label, 0))