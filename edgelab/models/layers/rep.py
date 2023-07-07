from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

# from ..base.general import ConvNormActivation,get_act
from edgelab.models.base.general import ConvNormActivation, get_act
from edgelab.registry import FUNCTIONS, MODELS


def padding_weights(
    weights: Optional[torch.Tensor] = None,
    shape: Union[int, Tuple[int, int]] = (3, 3),
    mode: str = 'constant',
    value: Union[int, float] = 0,
) -> Union[torch.Tensor, int]:
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
            'Wrong shape type, its type should be "int", "tuple", or "list",but got the type of {}'.format(type(shape))
        )

    if weights is None:
        return 0
    else:
        _, _, H, W = weights.shape
        assert (
            H < shape[0] and W < shape[1]
        ), "The size to be filled cannot be smaller than the shape size of the original weight, the original \
                size is {}, and the size to be filled is {}".format(
            (H, W), (shape)
        )
        return torch.nn.functional.pad(
            weights,
            ((shape[0] - W) // 2, (shape[0] - W) // 2, (shape[0] - H) // 2, (shape[0] - H) // 2),
            mode=mode,
            value=value,
        )


def fuse_conv_norm(
    block: Union[nn.Sequential, nn.BatchNorm2d, nn.LayerNorm], groups: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        conv_weight, conv_bias, norm_mean, norm_var, norm_gamm, norm_beta, norm_eps = (
            block.conv.weight,
            block.conv.bias,
            block.norm.running_mean,
            block.norm.running_var,
            block.norm.weight,
            block.norm.bias,
            block.norm.eps,
        )

        std = (norm_var + norm_eps).sqrt()
        t = (norm_gamm / std).reshape(-1, 1, 1, 1)
        return conv_weight * t, norm_beta + ((0 if conv_bias is None else conv_bias) - norm_mean) * norm_gamm / std
    elif isinstance(block, nn.BatchNorm2d):
        in_channels = block.num_features
        b = in_channels // groups
        norm_mean, norm_var, norm_gamm, norm_beta, norm_eps = (
            block.running_mean,
            block.running_var,
            block.weight,
            block.bias,
            block.eps,
        )

        w = torch.zeros((in_channels, b, 1, 1), dtype=torch.float32, device=norm_gamm.device)
        for idx in range(in_channels):
            w[idx, idx % b, 0, 0] = 1.0

        std = (norm_var + norm_eps).sqrt()
        norm_weight = (norm_gamm / std).reshape(-1, 1, 1, 1)
        return w * norm_weight, norm_beta - norm_gamm * norm_mean / std
    else:
        raise TypeError("Fusion module type should be Sequential or BatchNorm2d, but got {} type".format(type(block)))


@MODELS.register_module(force=True)
class RepConv1x1(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_res: bool = True,
        use_dense: bool = True,
        stride: int = 1,
        depth: int = 6,
        act_cfg: dict = dict(type="LeakyReLU"),
        init_cfg: Union[dict, List[dict], None] = None,
    ):
        super().__init__(init_cfg)

        self.depth = depth
        self.use_res = use_res
        self.use_dense = use_dense

        if stride > 1:
            self.down_sample = nn.MaxPool2d(2, stride=2, padding=0)
        else:
            self.down_sample = nn.Identity()

        self.conv3x3 = ConvNormActivation(in_channels, out_channels, 3, 1, 1, bias=True, activation_layer=None)
        self.conv = nn.ModuleList()

        for i in range(depth):
            layer = ConvNormActivation(out_channels, out_channels, 1, 1, 0, bias=True, activation_layer=None)
            self.conv.append(layer)
        self.norm = nn.ModuleList()
        if use_res:
            for i in range(depth):
                layer = nn.BatchNorm2d(out_channels)
                self.norm.append(layer)

        self.dense_norm = nn.BatchNorm2d(out_channels)

        self.fuse_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=1, bias=True)
        self.act = MODELS.build(act_cfg)

    def forward(self, x) -> None:
        x = self.down_sample(x)
        if self.training:
            x = self.conv3x3(x)
            if self.use_dense:
                dense_feature = self.dense_norm(x)
            for idx, layer in enumerate(self.conv):
                if self.use_res or (self.use_dense and idx > 0):
                    y = None
                    if self.use_res:
                        y = self.norm[idx](x)
                    if self.use_dense and idx > 0:
                        if y is not None:
                            y += dense_feature
                        else:
                            y = dense_feature + 0
                    x = layer(x) + y
                else:
                    x = layer(x)

        else:
            x = self.fuse_conv(x)

        x = self.act(x)
        return x

    def fuse1x1(self) -> torch.Tensor:
        weight, bias = fuse_conv_norm(self.conv[0])
        if self.use_res:
            norm_weight, norm_bias = fuse_conv_norm(self.norm[0])
            weight += norm_weight
            bias += norm_bias
        if len(self.conv) > 1:
            for idx, layer in enumerate(self.conv[1:]):
                weight_, bias_ = fuse_conv_norm(layer)
                if self.use_res:
                    norm_weight, norm_bias = fuse_conv_norm(self.norm[idx + 1])
                    weight_ += norm_weight
                    bias_ += norm_bias
                    weight = nn.functional.conv2d(weight.transpose(0, 1), weight_).transpose(0, 1)
                    bias = bias_ + (bias.view(1, -1, 1, 1) * weight_).sum((3, 2, 1))
                else:
                    weight = nn.functional.conv2d(weight.transpose(0, 1), weight_).transpose(0, 1)
                    bias = bias_ + (bias.view(1, -1, 1, 1) * weight_).sum((3, 2, 1))
                if self.use_dense:
                    norm_weight, norm_bias = fuse_conv_norm(self.dense_norm)
                    weight += norm_weight
                    bias += norm_bias

        return weight, bias

    def fuse_3x3_1x1(
        self, weight1: torch.Tensor, bias1: torch.Tensor, weight2: torch.Tensor, bias2: torch.Tensor
    ) -> torch.Tensor:
        weight = nn.functional.conv2d(weight1.transpose(0, 1), weight2).transpose(0, 1)
        bias = bias2 + (weight2 * bias1.view(1, -1, 1, 1)).sum((1, 2, 3))
        return weight, bias

    def rep(self):
        weight1x1, bias1x1 = self.fuse1x1()
        weight3x3, bias3x3 = fuse_conv_norm(self.conv3x3)
        w1, b1 = self.fuse_3x3_1x1(weight3x3, bias3x3, weight1x1, bias1x1)

        self.fuse_conv.weight.data = w1
        self.fuse_conv.bias.data = b1

    def train(self, mode: bool = True):
        res = super().train(mode)
        if not mode:
            self.rep()
        return res


@FUNCTIONS.register_module(force=True)
class Activation(nn.ReLU):
    """
    Series informed activation from https://arxiv.org/abs/2305.12972

    Params:
        in_channels(int): Number of channels of input
        act_num(int): Set the convolution kernel size(act*2+1) and set the
        padding size according to this value

    """

    def __init__(self, in_channels: int, act_num: int = 3) -> None:
        super().__init__()
        self.input_channels = in_channels
        self.act_num = act_num
        self.weight = torch.nn.Parameter(torch.randn(in_channels, 1, act_num * 2 + 1, act_num * 2 + 1))
        self.bn = nn.BatchNorm2d(in_channels, eps=1e6)
        self.bias = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            y = super().forward(x)
            y = nn.functional.conv2d(y, self.weight, self.bias, padding=self.act_num, groups=self.input_channels)
            return self.bn(y)
        else:
            y = super().forward(x)
            return nn.functional.conv2d(y, self.weight, self.bias, padding=self.act_num, groups=self.input_channels)

    def rep(self) -> None:
        norm_mean = self.bn.running_mean
        norm_var = self.bn.running_var
        norm_gamm = self.bn.weight
        norm_beta = self.bn.bias
        norm_eps = self.bn.eps

        std = (norm_var + norm_eps).sqrt()
        norm_weight = (norm_gamm / std).reshape(-1, 1, 1, 1)
        weight, bias = norm_weight * self.weight, norm_beta - norm_gamm * norm_mean / std
        self.weight.data.copy_(weight)
        self.bias = torch.nn.Parameter(torch.zeros(self.input_channels))
        self.bias.data.copy_(bias)

    def train(self, mode: bool = True):
        res = super().train(mode)
        if not mode:
            self.rep()
        return res


@MODELS.register_module(force=True)
class VanillaBlock(nn.Module):
    """
    VanillaNet Block from https://arxiv.org/abs/2305.12972

    Params:
        in_channels(int): Number of channels of input
        out_channels(int): Number of output channels
        act_num(int): Set the activation function convolution kernel size(act*2+1)
            and set the padding size according to this value
        stride(int): Step size during pooling
        add_pool(int): Whether to use global average pooling

    """

    def __init__(
        self, in_channels: int, out_channels: int, act_num: int = 3, stride: int = 1, add_pool: Optional[int] = None
    ) -> None:
        super().__init__()
        norm_config = dict(type='BatchNorm2d', eps=1e-6)
        self.act_learn = 1
        self.fused_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.conv1 = ConvNormActivation(
            in_channels, in_channels, 1, bias=True, norm_layer=norm_config, activation_layer=None
        )

        self.conv2 = ConvNormActivation(
            in_channels, out_channels, 1, bias=True, norm_layer=norm_config, activation_layer=None
        )
        if add_pool:
            self.pool = nn.Identity() if stride == 1 else nn.AdaptiveMaxPool2d((add_pool, add_pool))
        else:
            self.pool = nn.Identity() if stride == 1 else nn.MaxPool2d(stride)
        self.act = Activation(out_channels, act_num)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or self.act_learn != 1:
            x = self.conv1(x)
            x = nn.functional.leaky_relu(x, self.act_learn)
            x = self.conv2(x)
        else:
            x = self.fused_conv(x)

        x = self.pool(x)
        x = self.act(x)
        return x

    def rep(self) -> None:
        weight1, bias1 = fuse_conv_norm(self.conv1)

        weight2, bias2 = fuse_conv_norm(self.conv2)

        weight = weight2.transpose(1, 3).matmul(weight1.squeeze(3).squeeze(2)).transpose(1, 3)

        bias = (bias1.view(1, -1, 1, 1) * weight2).sum(3).sum(2).sum(1) + bias2

        self.fused_conv.weight.data.copy_(weight)
        self.fused_conv.bias.data.copy_(bias)

    def set_act_lr(self, lr: float) -> None:
        self.act_learn = lr

    def train(self, mode: bool = True):
        res = super().train(mode)
        if not mode:
            self.rep()
        return res


@MODELS.register_module(force=True)
class RepLine(BaseModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.kernel_size = kernel_size


@MODELS.register_module(force=True)
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

    def __init__(
        self,
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
        init_cfg: Union[dict, List[dict], None] = None,
    ) -> None:
        super().__init__(init_cfg)

        # if use_norm:
        #     assert in_channels == out_channels, "If you use norm, the number of input channels must be equal to the number of output channels,\
        #         but the input channel is {}, and the output channel is {}".format(
        #         in_channels, out_channels)
        self.use_norm = use_norm
        self.act_layer = act_layer
        self.kernel_size = kernel_size
        self.groups = groups
        self.conv_norm1 = ConvNormActivation(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
            norm_layer=norm_layer,
            activation_layer=None,
        )
        self.conv_norm2 = ConvNormActivation(
            in_channels,
            in_channels,
            1,
            stride=stride,
            padding=0,
            dilation=dilation,
            bias=bias,
            groups=groups,
            norm_layer=norm_layer,
            activation_layer=None,
        )
        if use_norm:
            self.norm = nn.BatchNorm2d(in_channels)
        if act_layer:
            self.act = get_act(act=act_layer)()
        if in_channels != out_channels:
            self.last_conv = ConvNormActivation(
                in_channels, out_channels, 1, stride=1, padding=0, norm_layer=norm_layer, activation_layer=act_layer
            )

        self.fused_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=True,
            padding_mode='zeros',
        )
        self.frist = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.use_norm:
                res = self.conv_norm1(x) + self.conv_norm2(x) + self.norm(x)
            else:
                res = self.conv_norm1(x) + self.conv_norm2(x)
            self.frist = True
        else:
            res = self.fused_conv(x)
        if self.act_layer:
            res = self.act(res)

        if hasattr(self, 'last_conv'):
            res = self.last_conv(res)

        return res

    def rep(self) -> None:
        kbn1, bbn1 = fuse_conv_norm(self.conv_norm1, self.groups)
        kbn2, bbn2 = fuse_conv_norm(self.conv_norm2, self.groups)

        weights, bias = kbn1 + padding_weights(kbn2), bbn1 + bbn2
        if self.use_norm:
            norm_weight, norm_bias = fuse_conv_norm(self.norm, self.groups)
            weights, bias = weights + padding_weights(norm_weight), norm_bias + bias

        try:
            self.fused_conv.weight.copy_(weights)
            self.fused_conv.bias.copy_(bias)
        except Exception:
            self.fused_conv.weight.data = weights
            self.fused_conv.bias.data = bias
            pass

    def train(self, mode: bool = True):
        res = super().train(mode)
        if not mode:
            self.rep()
        return res


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    rep = RepConv1x1(64, 32)
    rep.eval()
    input = torch.rand(1, 64, 192, 192)
    pred1 = rep(input, True)
    print("pred1::", pred1.shape)
    rep.eval()
    pred2 = rep(input, False)
    print("pred2::", pred2.shape)
    i2 = input
    var = (pred1 - pred2).abs().sum()
    print(var)
