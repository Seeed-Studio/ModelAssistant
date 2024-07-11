from torch import nn
import torch
from model.Block.ConvBlock import Conv_block1D, Conv_block2D
import torch.nn.functional as F


class up_sample(nn.Module):

    def __init__(self, in_channel, out_channel, tag):
        super().__init__()
        if tag == "Conv_block1D":
            self.down_sample = nn.Sequential(nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2, padding=0))
        elif tag == "Conv_block2D":
            self.down_sample = nn.Sequential(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        x = self.down_sample(x)
        return x


class Vae_Decode(nn.Module):

    def __init__(self, x_size, out_channel, in_channel, tag):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.Conv_block = tag
        latent_dim = 64
        middle_dim = 32
        self.x_size = int(x_size / 4)
        self.fc1 = nn.Linear(latent_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, self.x_size * self.x_size * out_channel)
        self.relu_x_fc1 = nn.ReLU()
        self.relu_x_fc2 = nn.ReLU()
        self.relu_c_fc1 = nn.ReLU()
        self.relu_c_fc2 = nn.ReLU()
        self.up_sample2 = up_sample(out_channel, out_channel, tag=self.Conv_block)
        self.conv1 = globals()[self.Conv_block](out_channel, out_channel)
        self.up_sample = up_sample(out_channel, out_channel, tag=self.Conv_block)
        self.conv2 = globals()[self.Conv_block](out_channel, out_channel)
        self.patch_debed = nn.ConvTranspose2d(out_channel, in_channel, kernel_size=1, stride=1, padding=0)
        self.conv_c_1 = globals()[self.Conv_block](out_channel, out_channel)
        self.up_sample_c = up_sample(out_channel, out_channel, tag=self.Conv_block)
        self.conv_c_2 = globals()[self.Conv_block](out_channel, in_channel)
        self.fc_c2 = nn.Linear(middle_dim, self.x_size * self.x_size * out_channel * 4)
        self.patch_debed_c = nn.ConvTranspose2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0)
        # self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, z1, z2):
        z = self.relu_x_fc1(self.fc1(z1))
        x = self.relu_x_fc2(self.fc2(z))
        x = x.contiguous().view(x.size(0), self.out_channel, self.x_size, self.x_size)
        x = self.up_sample2(x)
        res = self.conv1(x)
        x = res + x
        x = self.up_sample(x)
        x = self.conv2(x)
        x = self.patch_debed(x)

        c = self.relu_c_fc1(self.fc1(z2))
        c = self.relu_c_fc2(self.fc_c2(c))
        c = c.contiguous().view(c.size(0), self.out_channel, self.x_size * 2, self.x_size * 2)
        res = self.conv_c_1(c)
        c = res + c
        c = self.up_sample_c(c)
        c = self.conv_c_2(c)
        c = self.patch_debed_c(c)
        return x, c
