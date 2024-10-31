from mmengine.config import read_base

with read_base():
    from .._base_.default_runtime import *

from sscma.models.detectors.anomaly import Vae_Model
from sscma.models import Vae_Encode, Vae_Decode, Conv_block2D
from torch.nn import MSELoss
from sscma.datasets import Microphone_dataset
from sscma.evaluation import MseMetric
from sscma.deploy.models.anomaly_infer import AnomalyInfer

dataset_type = Microphone_dataset

data_root = ""

imgsz = (32, 32)
batch_size = 1

model = model = dict(
    type=Vae_Model,
    encode=dict(
        type=Vae_Encode, x_size=32, in_channel=3, out_channel=8, conv=Conv_block2D
    ),
    decode=dict(
        type=Vae_Decode,
        x_size=32,
        out_channel=8,
        in_channel=3,
        conv=Conv_block2D,
    ),
    loss=dict(type=MSELoss),
    freeze_randn=None,
)


deploy = dict(type=AnomalyInfer)


train_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=True,
    # collate_fn=dict(type=default_collate),
    dataset=dict(type=dataset_type, data_root=data_root),
)
val_dataloader = dict(
    batch_size=batch_size,
    num_workers=1,
    persistent_workers=True,
    drop_last=True,
    # collate_fn=dict(type=default_collate),
    dataset=dict(type=dataset_type, data_root=data_root),
)
test_dataloader = val_dataloader

train_cfg = dict(by_epoch=True, max_epochs=100)
val_cfg = dict()
test_cfg = dict()

val_evaluator = dict(type=MseMetric)
test_evaluator = val_evaluator

from torch.optim import Adam

optim_wrapper = dict(optimizer=dict(type=Adam, lr=1e-4, weight_decay=5e-4))
