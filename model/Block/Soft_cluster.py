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


class CustomLayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / (std + self.eps)
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias
        return x_normalized


def cdist(x1, x2):
    """
    计算两个输入张量之间的成对欧氏距离。

    参数:
    x1: Tensor of shape (m, d)
    x2: Tensor of shape (n, d)

    返回:
    dists: Tensor of shape (m, n)
    """
    # 检查输入张量的形状
    assert x1.shape[1] == x2.shape[1], "The number of dimensions must be the same for both inputs"

    # 计算平方和
    x1_square_sum = (x1 * x1).sum(dim=1, keepdim=True)
    x2_square_sum = (x2 * x2).sum(dim=1, keepdim=True)

    # 计算成对欧氏距离
    dists = torch.sqrt(x1_square_sum + x2_square_sum.permute(1, 0) - 2 * x1 @ x2.permute(1, 0))

    return dists


def torch_min_by_index(x, index):
    index = index.view(-1)
    x_min = [x[i, index[i]] for i in range(len(x))]
    x_min = torch.tensor(x_min).unsqueeze(0).unsqueeze(-1)
    return x_min


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

        x_min_index = torch.argmin(x, self.dims, keepdim=True)
        x_min = torch_min_by_index(x, x_min_index)
        exp_x = torch.exp((-self.alpha) * (x - x_min))  # 这是一个类似于紧缩的方法
        soft_x = exp_x / (exp_x.sum(self.dims, keepdim=True))

        return soft_x


class EuclidDistance_Assign_Module(nn.Module):

    def __init__(self, feature_dim, cluster_num=256, maxpool=1, soft_assign_alpha=32.0, is_grad=True):
        super(EuclidDistance_Assign_Module, self).__init__()
        # self.euclid_dis = torch.cdist
        self.act = nn.Sigmoid()
        self.feature_dim = feature_dim
        self.cluster_num = cluster_num
        self.norm = CustomLayerNorm(feature_dim)

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
        soft_assign = cdist(x, self.cluster_center)
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
