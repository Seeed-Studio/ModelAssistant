import torch
from torch import nn
from model.Block.ConvBlock import Conv_block1D, Conv_block2D
from einops import rearrange
import torch.nn.functional as F
from model.Block.Soft_cluster import EuclidDistance_Assign_Module, Space_EuclidDistance_Assign_Module


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


class Vae_Encode(nn.Module):

    def __init__(self, x_size, in_channel, out_channel, tag):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.Conv_block = tag
        latent_dim = 64
        middle_dim = 32
        self.patch_embed = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = globals()[self.Conv_block](out_channel, out_channel)
        self.down_sample = Down_sample(out_channel, out_channel, tag=self.Conv_block)
        self.conv2 = globals()[self.Conv_block](out_channel, out_channel)
        self.down_sample2 = Down_sample(out_channel, out_channel, tag=self.Conv_block)
        self.x_size = int(x_size / 4)
        self.fc1 = nn.Linear(self.x_size * self.x_size * out_channel, middle_dim)
        self.fc2_mu = nn.Linear(middle_dim, latent_dim)
        self.fc2_logvar = nn.Linear(middle_dim, latent_dim)
        self.relu_x_fc1 = nn.ReLU()
        self.relu_x_fc2_mu = nn.ReLU()
        self.relu_x_fc2_log = nn.ReLU()
        self.relu_c_fc1 = nn.ReLU()
        self.relu_c_fc2_mu = nn.ReLU()
        self.relu_c_fc2_log = nn.ReLU()
        self.patch_embed_c = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv1_c = globals()[self.Conv_block](out_channel, out_channel)
        self.down_sample_c = Down_sample(out_channel, out_channel, tag=self.Conv_block)
        self.conv2_c = globals()[self.Conv_block](out_channel, out_channel)
        self.x_size = int(x_size / 2)
        self.fc_c1 = nn.Linear(self.x_size * self.x_size * out_channel, middle_dim)
        self.cluster = EuclidDistance_Assign_Module(feature_dim=latent_dim, cluster_num=16, soft_assign_alpha=5)
        self.cluster2 = EuclidDistance_Assign_Module(feature_dim=latent_dim, cluster_num=32, soft_assign_alpha=5)

    def forward(self, x, c):
        x = self.patch_embed(x)
        x = self.conv1(x)
        x = self.down_sample(x)
        res = self.conv2(x)
        x = x + res
        x = self.down_sample2(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.relu_x_fc1(self.fc1(x))

        mu = self.relu_x_fc2_mu(self.fc2_mu(x))
        logvar = self.relu_x_fc2_log(self.fc2_logvar(x))
        mu_distance, mu_assign, mu = self.cluster(mu)
        cluster_mu = mu_distance * mu_assign
        log_distance, log_assign, logvar = self.cluster2(logvar)
        cluster_log = log_distance * log_assign

        c = self.patch_embed_c(c)
        c = self.conv1_c(c)
        c = self.down_sample_c(c)
        res = self.conv2_c(c)
        c = c + res
        c = c.contiguous().view(c.size(0), -1)
        c = self.relu_c_fc1(self.fc_c1(c))

        mu_c = self.relu_c_fc2_mu(self.fc2_mu(c))
        logvar_c = self.relu_c_fc2_log(self.fc2_logvar(c))

        return mu, mu_c, logvar, logvar_c, cluster_mu, cluster_log
