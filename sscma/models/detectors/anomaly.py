import torch
from torch import nn

from mmengine import MODELS
from mmengine.model import BaseModel


class Vae_Model(BaseModel):
    def __init__(
        self,
        encode: dict,
        decode: dict,
        loss: dict,
        freeze_randn=None,
    ):
        super().__init__()
        self.freeze_randn = freeze_randn
        if freeze_randn is not None:
            self.freeze_randn = nn.Parameter(self.freeze_randn, requires_grad=False)
            self.register_parameter("freeze_randn", self.freeze_randn)
        self.loss = nn.MSELoss()
        self.encode = MODELS.build(encode)
        self.decode = MODELS.build(decode)
        self.loss = MODELS.build(loss)

    def loss_function(self, recon_x, recon_c, x, c, mu, mu_c, logvar, logvar_c):
        BCE1 = self.loss(recon_x, x)
        BCE2 = self.loss(recon_c, c)
        KLD = -0.5 * torch.sum(1 + logvar - mu * mu - logvar.exp())
        KLD2 = -0.5 * torch.sum(1 + logvar_c - mu_c * mu_c - logvar_c.exp())
        return BCE1, BCE2, KLD + KLD2

    def _forward(self, x, c):
        mu, mu_c, logvar, logvar_c, res_x1, res_x2, res_x3 = self.encode(x, c)

        z = self.reparameterize(mu, logvar)
        c = self.reparameterize(mu_c, logvar_c)
        x, c = self.decode(z, c, res_x1, res_x2, res_x3)

        return x, c, mu, mu_c, logvar, logvar_c  # , cluster_loss

    def forward(self, batch):
        x = batch[:1, ...].clone()
        c = batch[1:2, ...].clone()

        mu, mu_c, logvar, logvar_c, res_x1, res_x2, res_x3 = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        c_recon = self.reparameterize(mu_c, logvar_c)
        x_recon, c_recon = self.decode(z, c_recon, res_x1, res_x2, res_x3)

        return x_recon, c_recon, x, c, mu, mu_c, logvar, logvar_c

    def train_step(self, data, optim_wrapper):
        res = self.training_step(data)
        optim_wrapper.update_params(res["loss"])
        return res

    def val_step(self, data):
        return self.validation_step(data)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        if self.freeze_randn is None:
            eps = torch.randn_like(std)
        else:
            eps = self.freeze_randn
        return mu + eps * std

    def training_step(self, batch):
        batch = batch[0].to(next(self.parameters()).device)
        x = batch[:1, ...]
        c = batch[1:2, ...]

        x_recon, c_recon, mu, mu_c, logvar, logvar_c = self._forward(x, c)

        loss1, loss2, loss3 = self.loss_function(
            x_recon, c_recon, x, c, mu, mu_c, logvar, logvar_c
        )
        loss = loss1 + loss2 + loss3  # + cluster_loss

        return dict(loss=loss)

    def validation_step(self, batch):
        batch = batch[0].to(next(self.parameters()).device)
        x = batch[:1, ...]
        c = batch[1:2, ...]

        x_recon, c_recon, mu, mu_c, logvar, logvar_c = self._forward(x, c)
        loss1, loss2, loss3 = self.loss_function(
            x_recon, c_recon, x, c, mu, mu_c, logvar, logvar_c
        )
        loss = loss1 + loss2 + loss3  # + cluster_loss
        return [dict(loss=loss)]
