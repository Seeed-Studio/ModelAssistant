import torch
from torch import nn
from typing import Callable


class CustomLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = torch.tensor(eps)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / (std + self.eps)
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized


class Down_sample(nn.Module):
    def __init__(self, in_channel, out_channel, tag):
        super().__init__()
        # self.expand_channel = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=1)
        if tag == "Conv_block1D":
            self.down_sample = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=0)
            )
        elif tag == "Conv_block2D":
            self.down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channel * 4, out_channel, kernel_size=3, stride=1, padding=1
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
            )

    def forward(self, x):
        # x = self.expand_channel(x)
        x0 = x[:, :, 0::2, 0::2]  # B H/2 W/2 C
        x1 = x[:, :, 1::2, 0::2]  # B H/2 W/2 C
        x2 = x[:, :, 0::2, 1::2]  # B H/2 W/2 C
        x3 = x[:, :, 1::2, 1::2]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], 1)  # B H/2 W/2 4*C
        x = self.down_sample(x)
        return x


class Vae_Encode(nn.Module):
    def __init__(self, x_size, in_channel, out_channel, conv: Callable):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        latent_dim = 64
        middle_dim = 32
        self.patch_embed = nn.Conv2d(
            in_channel, in_channel, kernel_size=3, stride=1, padding=1
        )
        self.conv1 = conv(in_channel, out_channel, layer_num=1)

        self.down_sample = Down_sample(out_channel, out_channel, tag=conv.__name__)
        self.conv2 = conv(out_channel, out_channel, layer_num=1)
        self.down_sample2 = Down_sample(out_channel, out_channel, tag=conv.__name__)
        self.x_size = int(x_size / 4)
        self.lnorm = nn.LayerNorm(self.x_size * self.x_size * out_channel)
        self.fc1 = nn.Linear(self.x_size * self.x_size * out_channel, middle_dim)
        self.fc2_mu = nn.Linear(middle_dim, latent_dim)
        self.fc2_logvar = nn.Linear(middle_dim, latent_dim)
        self.relu_x_fc1 = nn.ReLU()

        self.relu_c_fc1 = nn.ReLU()

        self.patch_embed_c = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv1_c = conv(out_channel, out_channel, layer_num=1)

        self.down_sample_c = Down_sample(out_channel, out_channel, tag=conv.__name__)
        self.conv2_c = conv(out_channel, out_channel, layer_num=1)
        self.x_size = int(x_size / 2)
        self.fc_c1 = nn.Linear(self.x_size * self.x_size * out_channel, middle_dim)

    def forward(self, x, c):
        x = self.patch_embed(x)
        res_x1 = x.clone()
        x = self.conv1(x)
        x = self.down_sample(x)
        res_x2 = x.clone()
        res = self.conv2(x)
        x = x + res
        x = self.down_sample2(x)
        res_x3 = x.clone()
        x = x.contiguous().view(x.size(0), -1)
        # x = self.lnorm(x)
        x = self.relu_x_fc1(self.fc1(x))
        mu = self.fc2_mu(x)
        logvar = self.fc2_logvar(x)

        c = self.patch_embed_c(c)
        c = self.conv1_c(c)
        c = self.down_sample_c(c)
        res = self.conv2_c(c)
        c = c + res

        c = c.contiguous().view(c.size(0), -1)
        c = self.relu_c_fc1(self.fc_c1(c))

        mu_c = self.fc2_mu(c)
        logvar_c = self.fc2_logvar(c)

        return (
            mu,
            mu_c,
            logvar,
            logvar_c,
            res_x1,
            res_x2,
            res_x3,
        )  # , cluster_mu, cluster_log
