from torch import nn
import torch
from model.Block.ConvBlock import Conv_block1D, Conv_block2D
import torch.nn.functional as F


class Signal_head(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.attn = nn.Linear(in_channel, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.attn(x)
        return x


class Sound_head(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # self.in_channel = in_channel
        # self.attn = nn.Sequential(
        #     nn.Linear(in_channel, in_channel),
        #     nn.ReLU(),
        #     nn.Linear(in_channel, in_channel),
        # )
        self.patch_debed = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        # for param in self.patch_debed.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        # B, C, H, W = x.size()
        # # x = rearrange(x, 'B C H W -> B (H W) C')
        # x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        # x = x.contiguous().view(B, H * W, C)  # (B, H*W, C)
        # x = self.attn(x)
        # # x = rearrange(x, 'B (H W) C -> B C H W', H = H, W=W)
        # x = x.view(B, H, W, self.in_channel)  # (B, H, W, C)
        # x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.patch_debed(x)
        return x


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


class Decode(nn.Module):

    def __init__(self, out_channel, in_channel, tag):
        super(Decode, self).__init__()
        Conv_block = tag
        self.conv1 = globals()[Conv_block](out_channel * 4, out_channel * 4)
        self.up_sample = up_sample(4 * out_channel, 2 * out_channel, tag=Conv_block)
        # self.process_layer.append(down_sample)
        self.dropout = nn.Dropout(p=0.5)
        self.conv2 = globals()[Conv_block](2 * out_channel, out_channel)
        if tag == "Conv_block1D":
            self.head = Signal_head(out_channel, in_channel)
            self.process = self.Signal_process
        elif tag == "Conv_block2D":
            self.head = Sound_head(out_channel, in_channel)
            self.process = self.Sound_process

    def Signal_process(self, x):
        x = x.permute(0, 2, 1)
        return x

    def Sound_process(self, x):
        return x

    def forward(self, x):
        x = self.process(x)
        res = self.conv1(x)
        x = x + res
        x = self.dropout(x)
        x = self.up_sample(x)
        x = self.conv2(x)
        x = self.head(x)
        return x


class Vae_Decode(nn.Module):

    def __init__(self, x_size, out_channel, in_channel, tag):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.Conv_block = tag
        latent_dim = 128
        middle_dim = 64
        self.x_size = int(x_size / 2)
        self.fc1 = nn.Linear(latent_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, self.x_size * self.x_size * out_channel)
        self.relu = nn.ReLU()
        self.conv1 = globals()[self.Conv_block](out_channel, out_channel)
        self.up_sample = up_sample(out_channel, out_channel, tag=self.Conv_block)
        self.conv2 = globals()[self.Conv_block](out_channel, out_channel)
        self.patch_debed = nn.ConvTranspose2d(out_channel, int(in_channel / 2), kernel_size=1, stride=1, padding=0)
        self.conv_c_1 = globals()[self.Conv_block](out_channel, out_channel)
        self.up_sample_c = up_sample(out_channel, out_channel, tag=self.Conv_block)
        self.conv_c_2 = globals()[self.Conv_block](out_channel, out_channel)
        self.fc_c2 = nn.Linear(middle_dim, self.x_size * self.x_size * out_channel)
        self.patch_debed_c = nn.ConvTranspose2d(out_channel, int(in_channel / 2), kernel_size=1, stride=1, padding=0)
        # self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.5)

    def forward(self, z1, z2):
        z = self.relu(self.fc1(z1))
        x = self.relu(self.fc2(z))
        x = x.view(x.size(0), self.out_channel, self.x_size, self.x_size)
        res = self.conv1(x)
        x = res + x
        x = self.up_sample(x)
        x = self.conv2(x)
        x = self.patch_debed(x)

        c = self.relu(self.fc1(z2))
        c = self.relu(self.fc_c2(c))
        c = c.view(c.size(0), self.out_channel, self.x_size, self.x_size)
        res = self.conv_c_1(c)
        c = res + c
        c = self.up_sample_c(c)
        c = self.conv_c_2(c)
        c = self.patch_debed_c(c)
        return x, c