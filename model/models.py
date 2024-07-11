import pytorch_lightning as pl
import torch
from torch import nn
import torch.optim as optim
from einops import rearrange
from model.Block.Encoder import Vae_Encode
from model.Block.Decoder import Vae_Decode
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from misc import tools


class Vae_Model(pl.LightningModule):

    def __init__(self, in_channel, out_channel, tag, freeze_randn=None):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.5))
        self.freeze_randn = freeze_randn
        if freeze_randn is not None:
            self.freeze_randn = nn.Parameter(self.freeze_randn, requires_grad=False)
            self.register_parameter('freeze_randn', self.freeze_randn)
        self.loss = nn.MSELoss()
        self.encode = Vae_Encode(x_size=32, in_channel=in_channel, out_channel=out_channel, tag=tag)
        self.decode = Vae_Decode(x_size=32, out_channel=out_channel, in_channel=in_channel, tag=tag)

    def psnr_loss(self, recon_x, recon_c, x, c):
        loss1 = tools.psnr(recon_x, x)
        loss2 = tools.psnr(recon_c, c)
        return loss1, loss2

    def loss_function(self, recon_x, recon_c, x, c, mu, mu_c, logvar, logvar_c):
        BCE1 = self.loss(recon_x, x)
        BCE2 = self.loss(recon_c, c)
        KLD = -0.5 * torch.sum(1 + logvar - mu * mu - logvar.exp())
        KLD2 = -0.5 * torch.sum(1 + logvar_c - mu_c * mu_c - logvar_c.exp())
        return BCE1, BCE2, KLD + KLD2

    def _forward(self, x, c):
        mu, mu_c, logvar, logvar_c, cluster_mu, cluster_log = self.encode(x, c)
        cluster_loss1 = torch.norm(cluster_mu, p=2) / 1000
        cluster_loss2 = torch.norm(cluster_log, p=2) / 1000
        cluster_loss = cluster_loss1 + cluster_loss2
        z = self.reparameterize(mu, logvar)
        c = self.reparameterize(mu_c, logvar_c)
        x, c = self.decode(z, c)
        return x, c, mu, mu_c, logvar, logvar_c, self.a, cluster_loss

    def forward(self, batch):
        # x = torch.cat((x, c), dim=1)
        x = batch[0:1, :].clone()
        c = batch[1:, :].clone()
        mu, mu_c, logvar, logvar_c, _, _ = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        c = self.reparameterize(mu_c, logvar_c)
        x, c = self.decode(z, c)
        return x, c

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.freeze_randn is None:
            eps = torch.randn_like(std)
        else:
            eps = self.freeze_randn
        return mu + eps * std

    def training_step(self, batch, batch_idx):
        # x = batch[:, :, :, :-1]
        x = batch[0]
        c = batch[1]
        # label = batch[2]
        x_recon, c_recon, mu, mu_c, logvar, logvar_c, a, cluster_loss = self._forward(x, c)
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
        x_recon, c_recon, mu, mu_c, logvar, logvar_c, a, cluster_loss = self._forward(x, c)
        loss1, loss2, loss3 = self.loss_function(x_recon, c_recon, x, c, mu, mu_c, logvar, logvar_c)
        loss = abs(a) * loss1 + abs(1 - a) * loss2 + loss3 + cluster_loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('3_loss', loss3, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        loss = self.loss_function(x, x_recon)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return [optimizer], [scheduler]
