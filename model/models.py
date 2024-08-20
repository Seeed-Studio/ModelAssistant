import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.optim as optim
from model.Block.Encoder import Vae_Encode
from model.Block.Decoder import Vae_Decode
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from misc import tools
import time


class Vae_Model(pl.LightningModule):

    def __init__(self, in_channel, out_channel, tag, freeze_randn=None):
        super().__init__()
        # self.a = nn.Parameter(torch.tensor(0.5))
        self.freeze_randn = freeze_randn
        if freeze_randn is not None:  # 若给定随机种子矩阵，则不再随机采样
            self.freeze_randn = nn.Parameter(self.freeze_randn, requires_grad=False)
            self.register_parameter('freeze_randn', self.freeze_randn)  # 注册随机种子矩阵参数
        self.loss = nn.MSELoss()
        self.encode = Vae_Encode(x_size=32, in_channel=in_channel, out_channel=out_channel, tag=tag)
        self.decode = Vae_Decode(x_size=32, out_channel=out_channel, in_channel=in_channel, tag=tag)

    def psnr_loss(self, recon_x, recon_c, x, c):
        loss1 = tools.psnr(recon_x, x)  # 推理时使用的psnr指标
        loss2 = tools.psnr(recon_c, c)
        return loss1, loss2

    def loss_function(self, recon_x, recon_c, x, c, mu, mu_c, logvar, logvar_c):
        BCE1 = self.loss(recon_x, x)  # 主loss函数
        BCE2 = self.loss(recon_c, c)
        KLD = -0.5 * torch.sum(1 + logvar - mu * mu - logvar.exp())  # 变分loss，仅辅助约束潜层特征的紧致性
        KLD2 = -0.5 * torch.sum(1 + logvar_c - mu_c * mu_c - logvar_c.exp())
        return BCE1, BCE2, KLD + KLD2

    def _forward(self, x, c):
        mu, mu_c, logvar, logvar_c, res_x1, res_x2, res_x3 = self.encode(x, c)
        # cluster_loss1 = torch.norm(cluster_mu, p=2) / 1000
        # cluster_loss2 = torch.norm(cluster_log, p=2) / 1000
        # cluster_loss = cluster_loss1 + cluster_loss2
        z = self.reparameterize(mu, logvar)
        c = self.reparameterize(mu_c, logvar_c)
        x, c = self.decode(z, c, res_x1, res_x2, res_x3)
        # k = x[0]
        # gadf_diff = np.transpose(k.detach().numpy(), (1, 2, 0))
        # image = (gadf_diff * 255).astype(np.uint8)
        # # 绘制结果
        # cv2.imshow('55', image)
        # cv2.waitKey(1)
        # time.sleep(0.01)
        return x, c, mu, mu_c, logvar, logvar_c  # , cluster_loss

    def forward(self, batch):
        # x = torch.cat((x, c), dim=1)
        x = batch[0:1, :].clone()
        c = batch[1:, :].clone()
        mu, mu_c, logvar, logvar_c, res_x1, res_x2, res_x3 = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        c = self.reparameterize(mu_c, logvar_c)
        x, c = self.decode(z, c, res_x1, res_x2, res_x3)
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
        x_recon, c_recon, mu, mu_c, logvar, logvar_c = self._forward(x, c)
        loss1, loss2, loss3 = self.loss_function(x_recon, c_recon, x, c, mu, mu_c, logvar, logvar_c)
        loss = loss1 + loss2 + loss3  # + cluster_loss
        self.log('loss1', loss1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('loss2', loss2, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # x = batch[:, :, :, :-1]
        x = batch[0]
        c = batch[1]
        # label = batch[2]
        x_recon, c_recon, mu, mu_c, logvar, logvar_c = self._forward(x, c)
        loss1, loss2, loss3 = self.loss_function(x_recon, c_recon, x, c, mu, mu_c, logvar, logvar_c)
        loss = loss1 + loss2 + loss3  # + cluster_loss
        self.log('val_loss', loss1, on_step=True, on_epoch=True, prog_bar=True)
        self.log('3_loss', loss3, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x = batch
        x_recon = self(x)
        loss = self.loss_function(x, x_recon)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        return [optimizer]  # , [scheduler]
