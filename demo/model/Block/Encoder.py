import torch
from torch import nn
from model.Block.ConvBlock import Conv_block1D, Conv_block2D
from einops import rearrange
import torch.nn.functional as F



class Signal_head(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.lstm = nn.LSTM(in_channel, out_channel, batch_first=True)
        self.attn = nn.Linear(out_channel, out_channel)

    def forward(self, x):
        # x = rearrange(x, 'B C T -> B T C')
        x = x.permute(0, 2, 1)
        x, (_, _) = self.lstm(x)
        x = self.attn(x)
        return x


class Sound_head(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            )


    def forward(self, x):
        B, C, H, W = x.size()
        # x = rearrange(x, 'B C H W -> B (H W) C')
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = x.contiguous().view(B, H * W, C)  # (B, H*W, C)
        x = self.attn(x)
        # x = rearrange(x, 'B (H W) C -> B C H W', H = H, W=W)
        x = x.view(B, H, W, C * 2)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class Down_sample(nn.Module):
    def __init__(self, in_channel, out_channel, tag):
        super().__init__()
        if tag == "Conv_block1D":
            self.down_sample = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
            )
        elif tag == "Conv_block2D":
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
            )

    def forward(self, x):
        x = self.down_sample(x)
        return x


class Encode(nn.Module):
    def __init__(self, in_channel, out_channel, tag):
        super(Encode, self).__init__()
        Conv_block = tag
        self.conv1 = globals()[Conv_block](in_channel, out_channel)
        self.down_sample = Down_sample(out_channel, 2 * out_channel, tag=Conv_block)
        self.conv2 = globals()[Conv_block](2*out_channel, 2*out_channel)
        if tag == "Conv_block1D":
            self.head = Signal_head(out_channel*2, out_channel*4)
            self.process = self.Signal_process
        elif tag == "Conv_block2D":
            self.head = Sound_head(out_channel*2, out_channel*4)
            self.process = self.Sound_process
    
    def Signal_process(self, x):
        # x = rearrange(x, 'B T C -> B C T')
        x = x.permute(0, 2, 1)
        return x

    def Sound_process(self, x):
        return x

    def forward(self, x):
        x = self.process(x)
        # x = self.Sound_process(x)
        x = self.conv1(x)
        x = self.down_sample(x)
        res = self.conv2(x)
        x = x + res
        x = self.head(x)
        return x