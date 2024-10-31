from torch import nn
import torch
from typing import Callable


class up_sample(nn.Module):
    def __init__(self, in_channel, out_channel, tag):
        super().__init__()
        if tag == "Conv_block1D":
            self.up_sample = nn.Sequential(
                nn.ConvTranspose1d(
                    in_channel, out_channel, kernel_size=3, stride=1, padding=1
                )
            )
        elif tag == "Conv_block2D":
            self.up_sample = nn.Sequential(
                nn.Conv2d(
                    in_channel, in_channel * 4, kernel_size=3, stride=1, padding=1
                )
            )
        self.trans = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.up_sample(x)
        x0 = x[:, 0:c, :, :].clone()
        x1 = x[:, c : 2 * c, :, :].clone()
        x2 = x[:, 2 * c : 3 * c, :, :].clone()
        x3 = x[:, 3 * c : 4 * c, :, :].clone()
        x = torch.reshape(x, (b, c, h * 2, w * 2))
        x[:, :, 0::2, 0::2] = x0  # B D H/2 W/2 C
        x[:, :, 1::2, 0::2] = x1  # B D H/2 W/2 C
        x[:, :, 0::2, 1::2] = x2  # B D H/2 W/2 C
        x[:, :, 1::2, 1::2] = x3  # B D H/2 W/2 C
        # x = x_temp
        x = self.trans(x)
        return x


from mmengine import MODELS


class Vae_Decode(nn.Module):

    def __init__(self, x_size, out_channel, in_channel, conv: Callable):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        latent_dim = 64
        middle_dim = 32
        self.x_size = int(x_size / 4)
        # self.lnorm = nn.LayerNorm(latent_dim)  # CustomLayerNorm(self.x_size * self.x_size * out_channel)
        self.fc1 = nn.Linear(latent_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, self.x_size * self.x_size * out_channel)
        self.relu_x_fc1 = nn.ReLU()
        self.relu_x_fc2 = nn.ReLU()
        self.relu_c_fc1 = nn.ReLU()
        self.relu_c_fc2 = nn.ReLU()
        self.up_sample2 = up_sample(out_channel, out_channel, tag=conv.__name__)

        self.conv1 = conv(out_channel, out_channel, layer_num=1)
        self.up_sample = up_sample(out_channel, out_channel, tag=conv.__name__)
        self.conv2 = conv(out_channel, in_channel, layer_num=1)
        self.patch_debed = nn.Conv2d(
            in_channel, in_channel, kernel_size=3, stride=1, padding=1
        )
        self.bn = nn.BatchNorm2d(in_channel)
        self.patch_debed2 = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0
        )
        self.conv_c_1 = conv(out_channel, out_channel, layer_num=1)
        self.up_sample_c = up_sample(out_channel, out_channel, tag=conv.__name__)
        self.conv_c_2 = conv(out_channel, in_channel, layer_num=1)
        self.fc_c2 = nn.Linear(middle_dim, self.x_size * self.x_size * out_channel * 4)
        self.patch_debed_c = nn.Conv2d(
            in_channel, in_channel, kernel_size=1, stride=1, padding=0
        )

    def forward(self, z1, z2, res_x1, res_x2, res_x3):
        z = self.relu_x_fc1(self.fc1(z1))
        x = self.relu_x_fc2(self.fc2(z))
        x = x.contiguous().view(x.size(0), self.out_channel, self.x_size, self.x_size)
        x = x + res_x3
        x = self.up_sample2(x)
        res = self.conv1(x)
        x = res + x
        x = x + res_x2
        x = self.up_sample(x)
        x = self.conv2(x)
        # x = x + res
        x = x + res_x1
        x = self.bn(self.patch_debed(x))
        x = self.patch_debed2(x)

        c = self.relu_c_fc1(self.fc1(z2))
        c = self.relu_c_fc2(self.fc_c2(c))
        c = c.contiguous().view(
            c.size(0), self.out_channel, self.x_size * 2, self.x_size * 2
        )
        res = self.conv_c_1(c)
        c = res + c
        c = self.up_sample_c(c)
        c = self.conv_c_2(c)
        c = self.patch_debed_c(c)
        return x, c
