from typing import List, Optional, Union, Tuple
import torch

import torch.nn as nn
from edgelab.registry import MODELS

from mmengine.model import BaseModule
from edgelab.models.base.general import ConvNormActivation, get_act


def padding_weights(weights: Optional[torch.Tensor] = None,
                    shape: Union[int, Tuple[int, int]] = (3, 3),
                    mode: str = 'constant',
                    value: Union[int, float] = 0) -> Union[torch.Tensor, int]:
    """
    Fill the convolution weights to the corresponding shape
    
    Params:
        weights: The weight value that needs to be filled
        shape: The size of the shape to fill,Default 3x3
        mode: fill pattern,``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
                Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``
    """
    if isinstance(shape, int):
        shape = (shape, shape)
    elif isinstance(shape, (tuple, list)):
        if len(shape) == 1:
            shape = (shape[0], shape[0])
    else:
        raise TypeError(
            'Wrong shape type, its type should be "int", "tuple", or "list",but got the type of {}'
            .format(type(shape)))

    if weights is None:
        return 0
    else:
        _, _, H, W = weights.shape
        assert H < shape[0] and W < shape[
            1], "The size to be filled cannot be smaller than the shape size of the original weight, the original \
                size is {}, and the size to be filled is {}".format((H, W),
                                                                    (shape))
        return torch.nn.functional.pad(
            weights, ((shape[0] - W) // 2, (shape[0] - W) // 2,
                      (shape[0] - H) // 2, (shape[0] - H) // 2),
            mode=mode,
            value=value)


def fuse_conv_norm(block: Union[nn.Sequential, nn.BatchNorm2d, nn.LayerNorm],
                   groups: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Fusion convolution and norm parameters, return weight and bias
    
    Params:
        block: Modules that require parameter fusion
        
    Return:
        weight: The weight after parameter fusion
        bias: The bias after parameter fusion
    '''
    block: nn.Sequential
    if isinstance(block, nn.Sequential):
        conv_weight, norm_mean, norm_var, norm_gamm, norm_beta, norm_eps = \
            block.conv.weight, block.norm.running_mean, block.norm.running_var, block.norm.weight, block.norm.bias, block.norm.eps

        std = (norm_var + norm_eps).sqrt()
        t = (norm_gamm / std).reshape(-1, 1, 1, 1)
        return conv_weight * t, norm_beta - norm_mean * norm_gamm / std
    elif isinstance(block, nn.BatchNorm2d):
        in_channels = block.num_features
        b = in_channels // groups
        norm_mean, norm_var, norm_gamm, norm_beta, norm_eps = \
            block.running_mean, block.running_var, block.weight, block.bias, block.eps

        w = torch.zeros((in_channels, b, 1, 1),
                        dtype=torch.float32,
                        device=norm_gamm.device)
        for idx in range(in_channels):
            w[idx, idx % b, 0, 0] = 1.0

        std = (norm_var + norm_eps).sqrt()
        norm_weight = (norm_gamm / std).reshape(-1, 1, 1, 1)
        return w * norm_weight, norm_beta - norm_gamm * norm_mean / std
    else:
        raise TypeError(
            "Fusion module type should be Sequential or BatchNorm2d, but got {} type"
            .format(type(block)))


@MODELS.register_module()
class RepLine(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)
        self.kernel_size = kernel_size


@MODELS.register_module()
class RepBlock(BaseModule):
    __doc__ = """
    Repeat parameter module: https://arxiv.org/abs/2101.03697
    
    Params:
        in_channels (int): Input feature map channel number
        out_channels (int): Output feature map channel number
        kernel_size (int or tuple[int, int]): Maximum convolution kernel size
        stride (int or tuple[int, int]): Stride of the convolution. Default: 1
        padding (int or tuple[int, int]):Padding added to all four sides of
            the input. Default: 1
        dilation (int or tuple[int,int]):Spacing between kernel elements. Default: 1
        bias (bool):If ``True``, adds a learnable bias to the
            output. Default: ``False`
        groups (int):Number of blocked connections from input
            channels to output channels. Default: 1
        norm_layer (str or dict): Norm layer configuration.Default: "BN"
        act_layer (str or dict): activation function.Default: None 
        use_norm (bool): Whether to connect a layer of BatchNorm in parallel.Default: False
        init_cfg (dict or list[dict] or None): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]] = 3,
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: int = 1,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 bias: bool = False,
                 groups: int = 1,
                 norm_layer: Union[str, dict] = "BN",
                 act_layer: Optional[str] = 'ReLU',
                 use_norm: bool = False,
                 init_cfg: Union[dict, List[dict], None] = None) -> None:
        super().__init__(init_cfg)

        # if use_norm:
        #     assert in_channels == out_channels, "If you use norm, the number of input channels must be equal to the number of output channels,\
        #         but the input channel is {}, and the output channel is {}".format(
        #         in_channels, out_channels)
        self.use_norm = use_norm
        self.act_layer = act_layer
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv_norm1 = ConvNormActivation(in_channels,
                                             in_channels,
                                             kernel_size,
                                             stride,
                                             padding=padding,
                                             dilation=dilation,
                                             bias=bias,
                                             groups=groups,
                                             norm_layer=norm_layer,
                                             activation_layer=None)
        self.conv_norm2 = ConvNormActivation(in_channels,
                                             in_channels,
                                             1,
                                             stride=stride,
                                             padding=0,
                                             dilation=dilation,
                                             bias=bias,
                                             groups=groups,
                                             norm_layer=norm_layer,
                                             activation_layer=None)
        if use_norm:
            self.norm = nn.BatchNorm2d(in_channels)
        if act_layer:
            self.act = get_act(act=act_layer)()
        if in_channels != out_channels:
            self.last_conv = ConvNormActivation(in_channels,
                                                out_channels,
                                                1,
                                                stride=1,
                                                padding=0,
                                                norm_layer=norm_layer,
                                                activation_layer=act_layer)

        self.conv3 = nn.Conv2d(in_channels,
                               in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding,
                               dilation=dilation,
                               groups=groups,
                               bias=True,
                               padding_mode='zeros')
        self.frist = True

    def forward(self, x):
        if self.training:
            if self.use_norm:
                res = self.conv_norm1(x) + self.conv_norm2(x) + self.norm(x)
            else:
                res = self.conv_norm1(x) + self.conv_norm2(x)
            self.frist = True
        else:
            res = self.conv3(x)
        if self.act_layer:
            res = self.act(res)

        if hasattr(self, 'last_conv'):
            res = self.last_conv(res)

        return res

    def rep(self):

        kbn1, bbn1 = fuse_conv_norm(self.conv_norm1, self.groups)
        kbn2, bbn2 = fuse_conv_norm(self.conv_norm2, self.groups)

        weights, bias = kbn1 + padding_weights(kbn2), bbn1 + bbn2
        if self.use_norm:
            norm_weight, norm_bias = fuse_conv_norm(self.norm, self.groups)
            weights, bias = weights + padding_weights(
                norm_weight), norm_bias + bias

        self.conv3.weight.data = weights
        self.conv3.bias.data = bias

    def eval(self):
        self.rep()
        return super().eval()

    def train(self, mode: bool = True):
        if not mode:
            self.rep()
        return super().train(mode)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    rep = RepBlock(32, 32, 3, use_norm=True)
    rep.eval()
    input = ii = torch.rand(4, 32, 192, 192)
    pred1 = rep(input)
    rep.eval()
    pred2 = rep(input)
    var = (pred1 - pred2).abs().sum()
    print(var)
