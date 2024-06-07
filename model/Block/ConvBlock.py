import torch
from torch import nn
import torch.nn.functional as F


class simam_module(torch.nn.Module):

    def __init__(self, channels=None, e_lambda=1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.GELU()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DepthwiseSeparableConv, self).__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)

        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class SelfAttention(nn.Module):

    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        out_channels = in_channels
        # Define the query, key, and value projections
        self.query = nn.Conv2d(out_channels, out_channels // 8, kernel_size=3, padding=1)
        self.key = nn.Conv2d(out_channels, out_channels // 8, kernel_size=3, padding=1)
        self.value = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm
        # Define the gamma parameter (learnable)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.patch_embed = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        # for param in self.patch_embed.parameters():
        #     param.requires_grad = False
        # self.patch_debed = nn.ConvTranspose2d(in_channels=out_channels, out_channels=in_channels, kernel_size=2, stride=2, padding=0)
        # for param in self.patch_debed.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        # x = self.patch_embed(x)
        batch_size, channels, height, width = x.size()

        # Apply the query, key, and value projections
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)

        # Compute attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)

        # Apply attention map to value

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # Apply the gamma parameter
        out = self.gamma * out + x
        # out = rearrange(out, "B C H W -> B H W C")
        # out = self.patch_debed(out)
        out = out.permute(0, 2, 3, 1)

        return out


class Conv_block1D(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv_block = nn.ModuleList()
        channel_list = [in_channel, out_channel]
        self.lstm = nn.LSTM(out_channel, out_channel, batch_first=True)
        for i in range(2):
            layer = nn.Sequential(nn.Conv1d(in_channels=channel_list[i], out_channels=out_channel, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.BatchNorm1d(out_channel))
            self.conv_block.append(layer)

    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
        x = x.permute(0, 2, 1)
        x, (_, _) = self.lstm(x)
        x = x.permute(0, 2, 1)
        return x


class Conv_block2D(nn.Module):

    def __init__(self, in_channel, out_channel, is_attn=False):
        super().__init__()
        self.conv_block = nn.ModuleList()
        channel_list = [in_channel, out_channel]
        for i in range(2):
            dilation = i + 1
            pad = dilation
            layer = nn.Sequential(
                nn.Conv2d(in_channels=channel_list[i], out_channels=out_channel, kernel_size=3, stride=1, padding=pad, dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm2d(out_channel),
            )
            self.conv_block.append(layer)
        if is_attn:
            self.attn = nn.Sequential(SelfAttention(out_channel), nn.LayerNorm(out_channel))
            # self.attn = nn.Sequential(simam_module())
        else:
            self.attn = None

    def forward(self, x):
        for layer in self.conv_block:
            x = layer(x)
            # if res.shape == x.shape:
            #     x = res + x
            # else:
            #     x = res

        if self.attn is not None:
            x = self.attn(x)
            x = x.permute(0, 3, 1, 2)
        return x
