from torch import nn
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
        self.out_channel = out_channel
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
        x = x.view(B, H, W, self.out_channel)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class up_sample(nn.Module):
    def __init__(self, in_channel, out_channel, tag):
        super().__init__()
        if tag == "Conv_block1D":
            self.down_sample = nn.Sequential(
                nn.ConvTranspose1d(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
            )
        elif tag == "Conv_block2D":
            self.down_sample = nn.Sequential(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
            )

    def forward(self, x):
        x = self.down_sample(x)
        return x


class Decode(nn.Module):
    def __init__(self, out_channel, in_channel, tag):
        super(Decode, self).__init__()
        Conv_block = tag
        self.conv1 = globals()[Conv_block](out_channel*4, out_channel*4)
        self.up_sample = up_sample(4*out_channel, 2 * out_channel, tag=Conv_block)
        # self.process_layer.append(down_sample)
        self.conv2 = globals()[Conv_block](2*out_channel, out_channel)
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
        # x = self.Sound_process(x)
        res = self.conv1(x)
        x = x + res
        x = self.up_sample(x)
        x = self.conv2(x)
        x = self.head(x)
        return x