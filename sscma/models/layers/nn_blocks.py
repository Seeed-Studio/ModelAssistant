from typing import List, Optional, Callable, AnyStr, Union, Dict
import math
import torch
import torch.nn as nn
from sscma.models.base.general import ConvNormActivation


def make_divisible(value, divisor: int, min_value=None, rounding_down_protect: bool = True) -> int:
    '''
    Make the input value divisible by a certain value

    Args:
        value: The input value
        divisor: The value to be divisible
        min_value: The minimum value
        rounding_down_protect: Whether to protect the rounding down
    Returns:
        The divisible value
    '''
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if rounding_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class MobileNetv4LayerScale(nn.Module):
    def __init__(self, init_value: float, channels, **kwargs) -> None:
        super().__init__(**kwargs)
        self._init_value = init_value

        self._params = nn.Parameter(
            torch.ones(channels, 1, 1, dtype=torch.float32) * init_value,
        )

    def forward(self, x):
        return x * self._params


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str = 'row') -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1.0 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class UniversalInvertedBottleneckBlock(nn.Module):
    '''
    An inverted bottleneck block with optional depthwises.
    '''

    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio,
        stride,
        activation: Union[Optional[Callable[..., nn.Module]], Dict, AnyStr] = nn.ReLU,
        normal_config: Union[Optional[dict], Dict, AnyStr] = {'type': 'BN', 'eps': 1e-3, 'momentum': 0.99},
        start_dw_kernel_size: int = 0,
        middle_dw_kernel_size: int = 3,
        middle_dw_downsample: bool = True,
        end_dw_kernel_size: int = 0,
        dw_activation: Union[Optional[Callable[..., nn.Module]], Dict, AnyStr] = None,
        division: int = 8,
        dialation: int = 1,
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        use_residual: bool = True,
        stochastic_depth_drop_rate: Optional[float] = None,
        output_intermediate_endpoints: bool = False,
    ):
        '''
        Initializes a UniversalInvertedBottleneckBlock.

        This is an extension of IB with optional depthwise convs before expansion (
        "starting" conv) and after projection ("ending" conv). Both of these convs
        are executed without activation. The standard depthwise conv of IB ("middle"
        conv) is optional too. This last one is followed by an activation, as in
        standard IBs. Squeeze-and-Excite or fused types of IBs are not supported.

        Args:
            in_channels: The number of input channels
            out_channels: The number of output channels
            expand_ratio: The expand ratio
            stride: The stride
            activation: The activation
            normal_config: The normal config
            start_dw_kernel_size: The kernel size of the starting depthwise conv
            middle_dw_kernel_size: The kernel size of the middle depthwise conv
            middle_dw_downsample: Whether to downsample the middle depthwise conv
            end_dw_kernel_size: The kernel size of the ending depthwise conv
            dw_activation: The activation of the depthwise conv
            division: The division
            dialation: The dialation
            use_layer_scale: Whether to use layer scale
            layer_scale_init_value: The layer scale init value
            use_residual: Whether to use residual
            stochastic_depth_drop_rate: The stochastic depth drop rate
            output_intermediate_endpoints: Whether to output intermediate endpoints

        '''
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._stride = stride
        self._expand_ratio = expand_ratio
        self._start_dw_kernel_size = start_dw_kernel_size
        self._middle_dw_kernel_size = middle_dw_kernel_size
        self._middle_dw_downsample = middle_dw_downsample
        self._end_dw_kernel_size = end_dw_kernel_size
        if not dw_activation:
            dw_activation = activation
        self._dw_activation = dw_activation
        self._use_layer_scale = use_layer_scale
        self._use_residule = use_residual
        self._normal_config = normal_config
        self._activation = activation
        self._division = division
        self._dialation = dialation
        self._layer_scale_init_value = layer_scale_init_value
        self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self._output_intermediate_endpoints = output_intermediate_endpoints

        self.layers = self.build_layer()

    def build_layer(self) -> nn.Sequential:
        '''
        All layers required to build this module
        '''
        blocks = []

        if self._start_dw_kernel_size:
            blocks.append(
                ConvNormActivation(
                    self._in_channels,
                    self._in_channels,
                    kernel_size=self._start_dw_kernel_size,
                    stride=self._stride if not self._middle_dw_downsample else 1,
                    groups=self._in_channels,
                    dilation=self._dialation,
                    bias=None,
                    padding=self._start_dw_kernel_size // 2,
                    activation_layer=None,
                    norm_layer=self._normal_config,
                )
            )

        expand_channles = make_divisible(self._in_channels * self._expand_ratio, self._division)
        blocks.append(
            ConvNormActivation(
                self._in_channels,
                expand_channles,
                kernel_size=1,
                stride=1,
                bias=None,
                norm_layer=self._normal_config,
                activation_layer=self._activation,
            )
        )

        if self._middle_dw_kernel_size:
            blocks.append(
                ConvNormActivation(
                    expand_channles,
                    expand_channles,
                    kernel_size=self._middle_dw_kernel_size,
                    stride=self._stride if self._middle_dw_downsample else 1,  # TODO
                    dilation=self._dialation,
                    bias=None,
                    padding=self._middle_dw_kernel_size // 2,
                    groups=expand_channles,
                    norm_layer=self._normal_config,
                    activation_layer=self._dw_activation,
                )
            )

        blocks.append(
            ConvNormActivation(
                expand_channles,
                self._out_channels,
                kernel_size=1,
                stride=1,
                bias=None,
                norm_layer=self._normal_config,
                activation_layer=None,
            )
        )

        if self._end_dw_kernel_size:
            blocks.append(
                ConvNormActivation(
                    self._out_channels,
                    self._out_channels,
                    kernel_size=self._end_dw_kernel_size,
                    stride=self._stride,  # TODO
                    dilation=self._dialation,
                    bias=False,
                    padding=self._end_dw_kernel_size // 2,
                    groups=self._out_channels,
                    norm_layer=self._normal_config,
                    activation_layer=None,
                )
            )

        if self._use_layer_scale:
            blocks.append(MobileNetv4LayerScale(self._layer_scale_init_value, self._out_channels))

        if self._use_residule and self._in_channels == self._out_channels and self._stride == 1:
            if self._stochastic_depth_drop_rate:
                blocks.append(StochasticDepth(self._stochastic_depth_drop_rate))

        return nn.Sequential(*blocks)

    def forward(self, inputs):
        endpoints = {}
        x = self.layers(inputs)

        if self._use_residule and self._in_channels == self._out_channels and self._stride == 1:
            x = x + inputs

        if self._output_intermediate_endpoints:
            return x, endpoints

        return x


class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        sequeeze_ratio,
        division=4,
        start_activation=nn.ReLU,
        end_activation=nn.Sigmoid,
        **kwargs,
    ) -> None:
        '''
        Creates a squeeze and excitation layer.

        Args:
            input_channels: int
            output_channels: int
            sequeeze_ratio: float
            division: int
            start_activation: nn.Module
            end_activation: nn.Module
            **kwargs: additional keyword arguments
        Returns:
            nn.Module
        '''
        super().__init__(**kwargs)
        middle_channels = make_divisible(max(1, input_channels * sequeeze_ratio), divisor=division)
        self.squeeze_conv = ConvNormActivation(
            input_channels,
            middle_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            activation_layer=start_activation,
            norm_layer=None,
        )

        self.squeeze_expand = ConvNormActivation(
            middle_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            activation_layer=end_activation,
            norm_layer=None,
        )

    def forward(self, inputs):
        inputs = torch.mean(inputs, dim=1, keepdim=True)
        x = self.squeeze_conv(inputs)
        x = self.squeeze_expand(x)
        return inputs * x


class InvertedBottleneckBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        expand_ratio,
        stride,
        kernel_size=3,
        dilation=1,
        activation=nn.ReLU,
        use_depthwise: bool = False,
        normal_config: Optional[dict] = {'type': 'BatchNorm2d', 'eps': 1e-6, 'momentum': 0.99},
        division: int = 1,
        squeeze_ratio: float = 0.25,
        squeeze_excitation: bool = True,
        use_layer_scale: bool = False,
        layer_scale_init_value: float = 1e-5,
        use_residual: bool = True,
        expand_se_in_filters=False,
        stochastic_depth_drop_rate: float = 0.0,
        output_intermediate_endpoints: bool = False,
        use_residule: bool = False,
        se_start_activation=nn.ReLU,
        se_end_activation=nn.Sigmoid,
        dw_activation=None,
        **kwargs,
    ):
        '''
        Initializes an inverted bottleneck block with BN after convolutions.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels
            expand_ratio: expand ratio of the block
            stride: stride of the block
            kernel_size: kernel size of the block
            dilation: dilation of the block
            activation: activation function of the block
            use_depthwise: whether to use depthwise convolution
            normal_config: configuration of the normalization layer
            division: division factor for the number of channels
            squeeze_ratio: ratio of squeeze channels
            squeeze_excitation: whether to use squeeze and excitation
            use_layer_scale: whether to use layer scale
            layer_scale_init_value: initial value of the layer scale
            use_residual: whether to use residual connection
            expand_se_in_filters: whether to expand the SE in filters
            stochastic_depth_drop_rate: stochastic depth drop rate
            output_intermediate_endpoints: whether to output intermediate endpoints
            use_residule: whether to use residual connection
            se_start_activation: activation function of the start of SE
            se_end_activation: activation function of the end of SE
            dw_activation: activation function of the depthwise convolution
            **kwargs: additional keyword arguments

        '''
        super().__init__()
        self._expand_ratio = expand_ratio
        self._use_depthwise = use_depthwise
        self._squeeze_excitation = squeeze_excitation
        self._squeeze_ratio = squeeze_ratio
        self._use_residule = use_residule
        self._stride = stride
        self._activation = activation
        self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self._output_intermediate_endpoints = output_intermediate_endpoints
        self._normal_config = normal_config
        self._division = division
        # expand_channels = make_divisible(in_channels * expand_ratio, divisor=division)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._kernel_size = kernel_size
        self._expand_se_in_filters = expand_se_in_filters
        self._se_start_activation = se_start_activation
        self._se_end_activation = se_end_activation
        self._dw_activation = dw_activation

        self.sequential = self.build_layer()

    def build_layer(self):
        '''
        All layers required to build this module


        '''
        layers = []
        expand_channels = make_divisible(self.in_channels * self._expand_ratio, self._division)

        expand_kernel = 1 if self._use_depthwise else self._kernel_size
        expand_stride = 1 if self._use_depthwise else self._stride
        # if self._expand_ratio > 1:
        #     layers.append(
        #         ConvNormActivation(
        #             self.in_channels,
        #             expand_channels,
        #             kernel_size=expand_kernel,
        #             stride=expand_stride,
        #             bias=False,
        #             padding=expand_kernel // 2,
        #             norm_layer=self._normal_config,
        #             activation_layer=self._activation,
        #         )
        #     )

        if self._use_depthwise:
            layers.append(
                ConvNormActivation(
                    self.in_channels,
                    expand_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    bias=False,
                    padding=self._kernel_size // 2,
                    # groups=expand_channels,
                    norm_layer=self._normal_config,
                    activation_layer=self._dw_activation,
                )
            )

        if self._squeeze_ratio and self._squeeze_ratio > 0 and self._squeeze_ratio <= 1:
            in_channels = expand_channels if self._expand_se_in_filters else self.in_channels
            layers.append(
                SqueezeExcitation(
                    in_channels,
                    expand_channels,
                    self._squeeze_ratio,
                    self._division,
                    self._se_start_activation,
                    self._se_end_activation,
                )
            )

        layers.append(
            ConvNormActivation(
                expand_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                padding=0,
                norm_layer=self._normal_config,
                activation_layer=None,
            )
        )

        if (
            self._use_residule
            and self.in_channels == self.out_channels
            and self._stride == 1
            and self._stochastic_depth_drop_rate
        ):
            layers.append(StochasticDepth(self._stochastic_depth_drop_rate))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        endpoints = {}
        shortcut = inputs

        x = self.sequential(inputs)

        if self._use_residule and self.in_channels == self.out_channels and self._stride == 1:
            x = x + shortcut
        if self._output_intermediate_endpoints:
            return x, endpoints

        return x


class OptimizedMultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_h_strides: int = 1,
        query_w_strides: int = 1,
        kv_strides: int = 1,
        dropout: float = 0,
        dw_kernel_size: int = 3,
        use_sync_bn: bool = False,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
        *args,
        **kwargs,
    ):
        ''' '''
        super().__init__()
        self._in_channels = in_channels
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim
        self._query_h_strides = query_h_strides
        self._query_w_strides = query_w_strides
        self._kv_strides = kv_strides
        self._dw_kernel_size = dw_kernel_size
        self._dropout = dropout
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        if use_sync_bn:
            self._normal = nn.SyncBatchNorm(in_channels)
        else:
            self._normal = nn.BatchNorm2d(in_channels)
        # self._layer = MultiHeadSelfAttentionBlock(*args, **kwargs)

    def build_layer(self):
        '''
        All layers required to build this module
        '''
        layers = []
        if self._query_h_strides > 1 or self._query_w_strides > 1:
            layers.append(nn.AvgPool2d(kernel_size=(self._query_h_strides, self._query_w_strides)))
            layers.append(self._normal)

        layers.append(
            nn.Conv2d(self._in_channels, self._num_heads * self._key_dim, kernel_size=1, stride=1, bias=False)
        )

        self.layer0 = nn.Sequential(*layers)

        layers = []

        if self._kv_strides > 1:
            layers.append(
                ConvNormActivation(
                    self._num_heads * self._key_dim,
                    self._num_heads * self._key_dim,
                    kernel_size=self._dw_kernel_size,
                    stride=self._kv_strides,
                    bias=False,
                    groups=self._num_heads * self._key_dim,
                    padding=self._dw_kernel_size // 2,
                    norm_layer=self._normal,
                    activation_layer=None,
                )
            )
        layers.append(nn.Conv2d(self._num_heads * self._key_dim, self._key_dim, kernel_size=1, stride=1, bias=False))
        self.layer1 = nn.Sequential(*layers)

        layers = []
        if self._kv_strides > 1:
            layers.append(
                ConvNormActivation(
                    self._key_dim,
                    self._key_dim,
                    kernel_size=self._dw_kernel_size,
                    stride=self._kv_strides,
                    bias=False,
                    groups=self._key_dim,
                    padding=self._dw_kernel_size // 2,
                    norm_layer=self._normal,
                    activation_layer=None,
                )
            )
        layers.append(nn.Conv2d(self._key_dim, self._value_dim, kernel_size=1, stride=1, bias=False))

        self.layer2 = nn.Sequential(*layers)

        layers = []

        if self._query_h_strides > 1 or self._query_w_strides > 1:
            layers.append(
                nn.Upsample(
                    scale_factor=(self._query_h_strides, self._query_w_strides), mode='bilinear', align_corners=False
                )
            )
        layers.append(nn.Conv2d(self._value_dim, self._in_channels, kernel_size=1, stride=1, bias=False))

        self.dropout = nn.Dropout(self._dropout)
        self.layer3 = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor):
        x = inputs
        B, C, H, W = inputs.shape

        q = self.layer0(x)
        x = q.reshape(B, H // self._query_h_strides * W // self._query_w_strides, self._num_heads, self._key_dim)

        k = self.layer1(x)

        num_elements = k.shape[2:].numel()
        channels = k.shape[1]
        k = k.reshape([B, channels, -1])

        logits = torch.einsum('bklh,bkp->bplh', q, k)

        logits = logits / math.sqrt(self._key_dim)
        attention_scores = self._dropout(torch.softmax(logits, dim=-1))

        v = self.layer2(x)

        channels = v.shape[1]
        v = v.reshape([B, channels, -1])

        output = torch.einsum('bplh,bkp->blhk', attention_scores, v)

        channels = output.shape[1]
        output = output.reshape(B, channels, H // self._query_h_strides, W // self._query_w_strides)

        output = self.layer3(output)

        output = output.reshape(x.shape)

        return output


class MultiQueryAttentionLayerV2(nn.Module):
    def __init__(self, num_heads=8, key_dim=64, value_dim=64, dropout=0.0, in_channels=64, *args, **kwargs):
        super().__init__()
        # self._layer = MultiHeadSelfAttentionBlock(*args, **kwargs)
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim
        self._dropout = dropout
        self._in_channels = in_channels
        self.build_layer()

    def build_layer(self):
        '''
        All layers required to build this module
        '''
        self.query = nn.Parameter(torch.zeros(self._num_heads, self._in_channels, self._key_dim, dtype=torch.float32))
        self.key = nn.Parameter(torch.zeros(self._in_channels, self._key_dim, dtype=torch.float32))
        self.value = nn.Parameter(torch.zeros(self._in_channels, self._value_dim, dtype=torch.float32))
        self.output = nn.Parameter(
            torch.zeros(self._in_channels, self._value_dim, self._num_heads, dtype=torch.float32)
        )
        self.dropout_layer = nn.Dropout(self._dropout)

    def reshape(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h * w)
        return x

    def forward(self, inputs):
        x, value = inputs
        x = self.reshape(x)
        value = self.reshape(value)

        q = torch.einsum('bdn,hdk->bknh', x, self.query)
        k = torch.einsum('bdm,dk->bkm', value, self.key)
        logits = torch.einsum('bknh,bkm->bmnh', q, k)

        logits = logits / torch.sqrt(torch.FloatTensor([self._key_dim]))

        logits = torch.softmax(logits, dim=-1)

        v = torch.einsum('bdm,dv->bvm', value, self.value)
        o = torch.einsum('bmnh,bvm->bvnh', logits, v)
        result = torch.einsum('bvnh,dvh->bdn', o, self.output)
        result = result.reshape_as(inputs[0])
        return result


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads=8,
        key_dim=64,
        value_dim=64,
        use_multi_query=False,
        query_h_strides=1,
        query_w_strides=1,
        kv_strides=1,
        downsampling_dw_kernel_size=3,
        dropout=0.0,
        use_bias=False,
        use_cpe=False,
        cpe_dw_kernel_size=7,
        stochastic_depth_drop_rate=None,
        use_residual=True,
        use_sync_bn=False,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        norm_momentum=0.99,
        norm_epsilon=0.001,
        output_intermediate_endpoints=False,
        *args,
        **kwargs,
    ) -> None:
        '''
        Initializes a MultiHeadSelfAttentionBlock.

        A Self-Attention block mixing tokens spatially and globally.

        Args:
            in_channels: number of input channels
            output_channels: number of output channels
            num_heads: number of heads
            key_dim: number of dimensions for the key
            value_dim: number of dimensions for the value
            use_multi_query: whether to use multi-query
            query_h_strides: stride of query in height
            query_w_strides: stride of query in width
            kv_strides: stride of key and value
            downsampling_dw_kernel_size: kernel size of the downsampling depthwise convolution
            dropout: dropout rate
            use_bias: whether to use bias
            use_cpe: whether to use convolutional positional encoding
            cpe_dw_kernel_size: kernel size of the convolutional positional encoding
            stochastic_depth_drop_rate: stochastic depth drop rate
            use_residual: whether to use residual connection
            use_sync_bn: whether to use sync batch normalization
            use_layer_scale: whether to use layer scale
            layer_scale_init_value: value of the layer scale
            norm_momentum: normalization momentum
            norm_epsilon: normalization epsilon
            output_intermediate_endpoints: whether to output intermediate endpoints
            *args: Additional positional arguments to be passed.
            **kwargs: Additional keyword arguments to be passed.

        '''
        super().__init__(*args, **kwargs)
        self._input_dim = in_channels
        self._output_channels = out_channels
        self._num_heads = num_heads
        self._key_dim = key_dim
        self._value_dim = value_dim
        self._use_multi_query = use_multi_query
        self._query_h_strides = query_h_strides
        self._query_w_strides = query_w_strides
        self._kv_strides = kv_strides
        self._downsampling_dw_kernel_size = downsampling_dw_kernel_size
        self._dropout = dropout
        self._use_bias = use_bias
        self._use_cpe = use_cpe
        self._cpe_dw_kernel_size = cpe_dw_kernel_size
        self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self._use_residual = use_residual
        self._use_sync_bn = use_sync_bn
        self._use_layer_scale = use_layer_scale
        self._layer_scale_init_value = layer_scale_init_value
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._output_intermediate_endpoints = output_intermediate_endpoints

        if self._use_cpe:

            self.conv0 = nn.Conv2d(
                self._input_dim,
                self._input_dim,
                kernel_size=self._cpe_dw_kernel_size,
                stride=1,
                padding=self._cpe_dw_kernel_size // 2,
                groups=self._input_dim,
                bias=True,
            )
        self.normal0 = nn.BatchNorm2d(self._input_dim, momentum=self._norm_momentum, eps=self._norm_epsilon)
        self.sequential = self.build_layer()

    def build_layer(self):
        '''
        All layers required to build this module
        '''
        layers = []

        if self._num_heads is None:
            self._num_heads = self._input_dim // self._key_dim

        if self._use_multi_query:
            if (self._query_h_strides or self._query_w_strides or self._kv_strides) > 1:
                layers.append(
                    OptimizedMultiQueryAttentionLayerWithDownSampling(
                        num_heads=self._num_heads,
                        key_dim=self._key_dim,
                        value_dim=self._value_dim,
                        query_h_strides=self._query_h_strides,
                        query_w_strides=self._query_w_strides,
                        kv_strides=self._kv_strides,
                        dw_kernel_size=self._downsampling_dw_kernel_size,
                        dropout=self._dropout,
                    )
                )
            else:
                layers.append(
                    MultiQueryAttentionLayerV2(
                        in_channels=self._input_dim,
                        num_heads=self._num_heads,
                        key_dim=self._key_dim,
                        value_dim=self._value_dim,
                        dropout=self._dropout,
                    )
                )
        else:
            layers.append(
                nn.MultiheadAttention(
                    num_heads=self._num_heads,
                    embed_dim=self._key_dim,
                    dropout=self._dropout,
                    bias=self._use_bias,
                )
            )

        if self._use_layer_scale:
            layers.append(
                MobileNetv4LayerScale(
                    channels=self._input_dim,
                    init_value=self._layer_scale_init_value,
                )
            )

        if self._use_residual:
            if self._stochastic_depth_drop_rate:
                layers.append(StochasticDepth(self._stochastic_depth_drop_rate))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        if self._use_cpe:
            inputs = self.conv0(inputs) + inputs

        x = self.normal0(inputs)
        if self._use_multi_query:
            if (self._query_h_strides or self._query_w_strides or self._kv_strides) > 1:
                x = self.sequential(x)
            else:
                x = self.sequential((x, x))
        # x = self.sequential(x)
        if self._use_residual:
            x = x + inputs

        return x


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
