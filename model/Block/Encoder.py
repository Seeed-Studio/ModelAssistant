import torch
from torch import nn
from model.Block.ConvBlock import Conv_block1D, Conv_block2D
from einops import rearrange
import torch.nn.functional as F
from model.Block.Soft_cluster import EuclidDistance_Assign_Module, Space_EuclidDistance_Assign_Module


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
        self.in_channel = in_channel
        self.out_channel = out_channel
        # self.attn = nn.Sequential(nn.Linear(in_channel, out_channel), nn.GELU(), nn.Linear(out_channel, out_channel), nn.GELU())

    def forward(self, x, x_diff):
        x = torch.cat((x, x_diff), dim=1)
        # B, C, H, W = x.size()
        # # x = rearrange(x, 'B C H W -> B (H W) C')
        # x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        # x = x.contiguous().view(B, H * W, C)  # (B, H*W, C)
        # x = self.attn(x)
        # # x = rearrange(x, 'B (H W) C -> B C H W', H = H, W=W)
        # x = x.view(B, H, W, self.out_channel)  # (B, H, W, C)
        # x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class Down_sample(nn.Module):

    def __init__(self, in_channel, out_channel, tag):
        super().__init__()
        if tag == "Conv_block1D":
            self.down_sample = nn.Sequential(nn.Conv1d(in_channel, out_channel, kernel_size=2, stride=2, padding=0))
        elif tag == "Conv_block2D":
            self.down_sample = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        x = self.down_sample(x)
        return x


class Encode(nn.Module):

    def __init__(self, in_channel, out_channel, tag):
        super(Encode, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.Conv_block = tag
        self.conv1 = globals()[self.Conv_block](out_channel, out_channel)
        # self.dropout = nn.Dropout(p=0.5)
        self.down_sample = Down_sample(out_channel, 2 * out_channel, tag=self.Conv_block)
        self.conv2 = globals()[self.Conv_block](2 * out_channel, 2 * out_channel)

        if tag == "Conv_block1D":
            self.head = Signal_head(out_channel * 2, out_channel * 4)
            self.process = self.Signal_process
        elif tag == "Conv_block2D":
            self.head = Sound_head(out_channel * 2, out_channel * 4)
            self.process = self.Sound_process

        # self.conv_diff_1 = globals()[self.Conv_block](out_channel, out_channel)
        self.diff_down_sample = Down_sample(out_channel, 2 * out_channel, tag=self.Conv_block)
        # 0elf.conv_diff_2 = globals()[self.Conv_block](2 * out_channel, 2 * out_channel)
        self.cluster = EuclidDistance_Assign_Module(feature_dim=out_channel * 4, cluster_num=4)
        self.sim_cluster = Space_EuclidDistance_Assign_Module(feature_dim=3, cluster_num=4, soft_assign_alpha=4096)

    def Signal_process(self, x):
        # x = rearrange(x, 'B T C -> B C T')
        x = x.permute(0, 2, 1)
        return x

    def Sound_process(self, x):
        patch_embed = nn.Conv2d(in_channels=self.in_channel, out_channels=self.out_channel, kernel_size=1, stride=1, padding=0)
        # for param in patch_embed.parameters():
        #     param.requires_grad = False
        x = patch_embed(x)
        return x

    # def self_similarity(self, x):
    #     x_sim = x  #  * x.permute(0, 1, 2, 3)
    #     sim_vector = torch.norm(x_sim, p=2, dim=(2, 3), keepdim=True)
    #     return sim_vector.permute(0, 2, 3, 1)

    def forward(self, x_diff, x, c):
        # sim_vector = self.self_similarity(x_diff)
        c = c.unsqueeze(-1).unsqueeze(-1)
        sim_distance, sim_assign = self.sim_cluster(c.permute(0, 2, 3, 1))
        sim_loss = torch.norm(sim_distance * sim_assign, p=2) / 1000

        x = self.process(x_diff)
        x = self.conv1(x)
        x = self.down_sample(x)
        res = self.conv2(x)
        # ker_distance, ker_assign, _ = self.ker_cluster(image_kernel)
        # ker_loss = torch.norm(ker_distance * ker_assign, p=2) / 10

        x = self.head(res, x)
        x = x.permute(0, 2, 3, 1)
        x_distance, x_assign, x = self.cluster(x)
        x = x.permute(0, 3, 1, 2)
        cluster_loss = torch.norm(x_distance * x_assign, p=2) / 1000
        # x = self.dropout(x)
        return x, cluster_loss, sim_loss


class Vae_Encode(nn.Module):

    def __init__(self, x_size, in_channel, out_channel, tag):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.Conv_block = tag
        latent_dim = 128
        middle_dim = 64
        self.patch_embed = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = globals()[self.Conv_block](out_channel, out_channel)
        self.down_sample = Down_sample(out_channel, out_channel, tag=self.Conv_block)
        self.conv2 = globals()[self.Conv_block](out_channel, out_channel)
        self.x_size = int(x_size / 2)
        self.fc1 = nn.Linear(self.x_size * self.x_size * out_channel, middle_dim)
        self.fc2_mu = nn.Linear(middle_dim, latent_dim)
        self.fc2_logvar = nn.Linear(middle_dim, latent_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.patch_embed_c = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv1_c = globals()[self.Conv_block](out_channel, out_channel)
        self.down_sample_c = Down_sample(out_channel, out_channel, tag=self.Conv_block)
        self.conv2_c = globals()[self.Conv_block](out_channel, out_channel)
        self.x_size = int(x_size / 2)
        self.fc_c1 = nn.Linear(self.x_size * self.x_size * out_channel, middle_dim)
        self.cluster = EuclidDistance_Assign_Module(feature_dim=latent_dim, cluster_num=32, soft_assign_alpha=2048)
        self.cluster2 = EuclidDistance_Assign_Module(feature_dim=latent_dim, cluster_num=32, soft_assign_alpha=2048)

    def forward(self, x, c):
        x = self.patch_embed(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.down_sample(x)
        res = self.conv2(x)
        x = x + res
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        c = self.patch_embed_c(c)
        c = self.conv1_c(c)
        c = self.dropout(c)
        c = self.down_sample_c(c)
        res = self.conv2_c(c)
        c = c + res
        c = self.dropout(c)
        c = c.view(c.size(0), -1)
        c = self.relu(self.fc_c1(c))
        c = self.dropout(c)

        mu = self.relu(self.fc2_mu(x))
        logvar = self.relu(self.fc2_logvar(x))
        mu_distance, mu_assign, mu = self.cluster(mu)
        cluster_loss = torch.norm(mu_distance * mu_assign, p=2) / 1000
        log_distance, log_assign, logvar = self.cluster2(logvar)
        cluster_loss2 = torch.norm(log_distance * log_assign, p=2) / 1000

        mu_c = self.relu(self.fc2_mu(c))
        logvar_c = self.relu(self.fc2_logvar(c))
        return mu, mu_c, logvar, logvar_c, cluster_loss + cluster_loss2
