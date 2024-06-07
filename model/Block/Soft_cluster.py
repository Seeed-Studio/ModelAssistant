import sys
import os
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from einops import rearrange


def cluster_alpha(max_n=40):
    constant_value = 1  # specs.embedding_size # Used to modify the range of the alpha scheme
    # max_n = 40  # Number of alpha values to consider
    alphas = np.zeros(max_n, dtype=float)
    alphas[0] = 0.1
    for i in range(1, max_n):
        alphas[i] = (2**(1 / (np.log(i + 1))**2)) * alphas[i - 1]
    alphas = alphas / constant_value
    # print(alphas)
    return alphas


class PosSoftAssign(nn.Module):

    def __init__(self, dims=1, alpha=1.0):
        super(PosSoftAssign, self).__init__()
        self.dims = dims
        self.alpha = alpha

    def forward(self, x, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        x_max, _ = torch.max(x, self.dims, keepdim=True)
        exp_x = torch.exp(self.alpha * (x - x_max))
        soft_x = exp_x / (exp_x.sum(self.dims, keepdim=True))
        return soft_x


class NegSoftAssign(nn.Module):

    def __init__(self, dims=1, alpha=32.0):
        super(NegSoftAssign, self).__init__()
        self.dims = dims
        self.alpha = alpha

    def forward(self, x, alpha=None):
        if alpha is not None:
            self.alpha = alpha

        x_min, _ = torch.min(x, self.dims, keepdim=True)
        exp_x = torch.exp((-self.alpha) * (x - x_min))  # 这是一个类似于紧缩的方法
        soft_x = exp_x / (exp_x.sum(self.dims, keepdim=True))
        return soft_x


class EuclidDistance_Assign_Module(nn.Module):

    def __init__(self, feature_dim, cluster_num=256, maxpool=1, soft_assign_alpha=32.0, is_grad=True):
        super(EuclidDistance_Assign_Module, self).__init__()
        self.euclid_dis = torch.cdist
        self.act = nn.Sigmoid()
        self.feature_dim = feature_dim
        self.cluster_num = cluster_num
        self.norm = nn.LayerNorm(feature_dim)

        self.assign_func = NegSoftAssign(-1, soft_assign_alpha)
        self.register_param(is_grad)

    def register_param(self, is_grad):
        cluster_center = nn.Parameter(torch.rand(self.cluster_num, self.feature_dim), requires_grad=is_grad)
        identity_matrix = nn.Parameter(torch.eye(self.cluster_num), requires_grad=False)
        self.register_parameter('cluster_center', cluster_center)
        self.register_parameter('identity_matrix', identity_matrix)
        return

    def self_similarity(self):

        return self.euclid_dis(self.cluster_center, self.cluster_center)

    def forward(self, x, alpha=None):
        #   传入x尺寸为B*D*H*W*C
        x_temp = x.clone()
        x = self.norm(x_temp)
        soft_assign = self.euclid_dis(x, self.cluster_center.unsqueeze(0))  # 这里返回的是向量间的距离
        x_distance = soft_assign
        x_distance_assign = self.assign_func(x_distance, alpha)
        kk = self.cluster_center.clone()
        x_rec = x_distance_assign @ kk

        return x_distance, x_distance_assign, x_rec.squeeze(0)


class Space_EuclidDistance_Assign_Module(nn.Module):

    def __init__(self, feature_dim, cluster_num=2, maxpool=1, soft_assign_alpha=32.0):
        super(Space_EuclidDistance_Assign_Module, self).__init__()
        self.euclid_dis = torch.cdist
        self.act = nn.Sigmoid()
        self.feature_dim = feature_dim
        self.cluster_num = cluster_num
        self.norm = nn.LayerNorm(feature_dim)
        self.assign_func = NegSoftAssign(-1, soft_assign_alpha)
        self.register_param()

    def register_param(self, ):
        cluster_center = nn.Parameter(torch.rand(self.cluster_num, self.feature_dim), requires_grad=True)
        ident_temp = torch.empty((self.feature_dim, self.cluster_num, self.cluster_num))
        for i in range(self.feature_dim):
            ident_temp[i] = torch.eye(self.cluster_num)
        identity_matrix = nn.Parameter(ident_temp, requires_grad=False)
        self.register_parameter('cluster_center', cluster_center)
        self.register_parameter('identity_matrix', identity_matrix)
        return

    def self_similarity(self):
        return self.euclid_dis(self.cluster_center, self.cluster_center)

    def forward(self, x, alpha=None):
        #   传入x尺寸为B*D*H*W*C
        # x_temp = x.clone()
        # x = self.norm(x_temp)
        B, H, W, C = x.shape
        x_re = rearrange(x, 'B H W C -> (H W) B C')
        soft_assign = self.euclid_dis(x_re.contiguous(), self.cluster_center.unsqueeze(0))  # 这里返回的是向量间的距离
        # 若是输入b*w*h*c的矩阵，返回则是b*w*h*1
        x_distance = rearrange(soft_assign, '(H W) B CN -> B CN H W', H=H)
        x_distance_assign = self.assign_func(x_distance, alpha)
        return x_distance, x_distance_assign
