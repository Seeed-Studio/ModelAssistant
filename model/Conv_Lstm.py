import pytorch_lightning as pl
import torch
from torch import nn
import torch.optim as optim
from einops import rearrange
from model.Block.Encoder import Encode
from model.Block.Decoder import Decode
import torch.nn.functional as F


# Define the LightningModule
class SimpleModel(pl.LightningModule):

    def __init__(self, in_channel, out_channel, tag):
        super(SimpleModel, self).__init__()
        self.encode = Encode(in_channel=in_channel, out_channel=out_channel, tag=tag)
        self.decode = Decode(out_channel=out_channel, in_channel=in_channel, tag=tag)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        loss = self.loss_function(x, x_recon)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        val_loss = self.loss_function(x, x_recon)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        loss = self.loss_function(x, x_recon)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
