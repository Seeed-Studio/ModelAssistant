from mmengine.optim import LinearLR
from mmengine.optim import CosineAnnealingLR
from torch.optim import AdamW

# optimizer
# for batch in each gpu is 256, 1 gpu
#  lr = 5e-5 * 256 / 512 = 2.5e-5
optim_wrapper = dict(
    optimizer=dict(type=AdamW, lr=2.5e-5, eps=1e-7, weight_decay=0.1),
    # specific to vit pretrain
    paramwise_cfg=dict(
        custom_keys={
            ".cls_token": dict(decay_mult=0.0),
            ".pos_embed": dict(decay_mult=0.0),
        }
    ),
)

# learning policy
warmup_epochs = 20
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type=LinearLR,
        start_factor=4e-4,
        by_epoch=True,
		begin=1,
        end=warmup_epochs,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(type=CosineAnnealingLR, eta_min=1e-8, by_epoch=True, begin=warmup_epochs),
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=256)
