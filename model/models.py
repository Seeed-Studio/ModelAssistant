import pytorch_lightning as pl
import torch
from torch import nn
import torch.optim as optim
from einops import rearrange
from model.Block.Encoder import Encode, Vae_Encode
from model.Block.Decoder import Decode, Vae_Decode
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
        # x = batch[:, :, :, :-1]
        x = batch
        x_recon = self(x)
        # loss = F.l1_loss(x[:, :, :3], x_recon)
        loss = self.loss_function(x, x_recon)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # x = batch[:, :, :, :-1]
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


class Double_stream_Model(pl.LightningModule):

    def __init__(self, in_channel, out_channel, tag):
        super().__init__()
        self.encode = Encode(in_channel=in_channel, out_channel=out_channel, tag=tag)
        self.decode = Decode(out_channel=out_channel, in_channel=in_channel, tag=tag)
        self.loss_function = nn.MSELoss()

    def forward(self, x_diff, x, signal_description):
        x, cluster_loss, ker_loss = self.encode(x_diff, x, signal_description)
        x = self.decode(x)
        return x, cluster_loss, ker_loss

    def training_step(self, batch, batch_idx):
        # x = batch[:, :, :, :-1]
        x_diff = batch[0]
        x = batch[1]
        label = batch[2]
        signal_description = batch[3]
        x_recon, cluster_loss, sim_loss = self(x_diff, x, signal_description)
        # loss = F.l1_loss(x[:, :, :3], x_recon)
        main_loss = self.loss_function(label, x_recon)
        loss = main_loss + cluster_loss + sim_loss
        self.log('t_loss', main_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('tc_loss', sim_loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # x = batch[:, :, :, :-1]
        x_diff = batch[0]
        x = batch[1]
        label = batch[2]
        signal_description = batch[3]
        x_recon, cluster_loss, sim_loss = self(x_diff, x, signal_description)
        main_loss = self.loss_function(label, x_recon) + sim_loss
        # val_loss = main_loss + cluster_loss
        self.log('val_loss', main_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('vc_loss', cluster_loss, on_step=True, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        loss = self.loss_function(x, x_recon)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer


class Vae_Model(pl.LightningModule):

    def __init__(self, in_channel, out_channel, tag):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
        self.encode = Vae_Encode(x_size=32, in_channel=in_channel, out_channel=out_channel, tag=tag)
        self.decode = Vae_Decode(x_size=32, out_channel=out_channel, in_channel=in_channel, tag=tag)

    def loss_function(self, recon_x, recon_c, x, c, mu, mu_c, logvar, logvar_c):
        BCE1 = F.mse_loss(recon_x, x, reduction='mean')
        BCE2 = F.mse_loss(recon_c, c, reduction='mean')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD2 = -0.5 * torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
        return BCE1, BCE2, KLD + KLD2

    def forward(self, x, c):
        # x = torch.cat((x, c), dim=1)
        mu, mu_c, logvar, logvar_c, cluster_loss = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        c = self.reparameterize(mu_c, logvar_c)
        x, c = self.decode(z, c)
        return x, c, mu, mu_c, logvar, logvar_c, self.a, cluster_loss

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch, batch_idx):
        # x = batch[:, :, :, :-1]
        x = batch[0]
        c = batch[1]
        # label = batch[2]
        x_recon, c_recon, mu, mu_c, logvar, logvar_c, a, cluster_loss = self(x, c)
        loss1, loss2, loss3 = self.loss_function(x_recon, c_recon, x, c, mu, mu_c, logvar, logvar_c)
        loss = abs(a) * loss1 + abs(1 - a) * loss2 + loss3 + cluster_loss
        self.log('loss1', loss1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss2', loss2, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # x = batch[:, :, :, :-1]
        x = batch[0]
        c = batch[1]
        # label = batch[2]
        x_recon, c_recon, mu, mu_c, logvar, logvar_c, a, cluster_loss = self(x, c)
        loss1, loss2, loss3 = self.loss_function(x_recon, c_recon, x, c, mu, mu_c, logvar, logvar_c)
        loss = abs(a) * loss1 + abs(1 - a) * loss2 + loss3 + cluster_loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        loss = self.loss_function(x, x_recon)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
