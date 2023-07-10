import tensorflow as tf
import torch.nn as nn
from tensorflow import keras


class TFBN(keras.layers.Layer):
    def __init__(self, w=None):
        super().__init__()
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.detach().numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.detach().numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.detach().numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.detach().numpy()),
            epsilon=w.eps,
            momentum=w.momentum,
        )

    def call(self, inputs):
        return self.bn(inputs)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)  # auto-pad
    return p


class TFPad(keras.layers.Layer):
    def __init__(self, pad):
        super().__init__()
        self._padding = pad
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])

    def call(self, inputs):
        if self._padding == 0:
            return tf.identity(inputs)
        else:
            return tf.pad(inputs, self.pad, mode='constant', constant_values=0)


class TFMaxPool2d(keras.layers.Layer):
    def __init__(self, w=None):
        super().__init__()
        pad = True if (w.stride == 1 and w.padding == w.kernel_size // 2) else False
        maxpool = keras.layers.MaxPool2D(
            w.kernel_size,
            w.stride,
            'SAME' if pad else 'VALID',
        )
        self.maxpool = maxpool if pad else keras.Sequential([TFPad(autopad(w.kernel_size, w.padding)), maxpool])

    def call(self, inputs):
        return self.maxpool(inputs)


class TFAvgPool2d(keras.layers.Layer):
    def __init__(self, w=None):
        super().__init__()
        pad = True if (w.stride == 1 and w.padding == w.kernel_size // 2) else False
        avgpool = keras.layers.AveragePooling2D(
            w.kernel_size,
            w.stride,
            'SAME' if pad else 'VALID',
        )
        self.maxpool = avgpool if pad else keras.Sequential([TFPad(autopad(w.kernel_size, w.padding)), avgpool])

    def call(self, inputs):
        return self.maxpool(inputs)


class TFBaseConv2d(keras.layers.Layer):
    # Standard convolution2d or depthwiseconv2d depends on 'g' argument.
    def __init__(self, w=None):
        super().__init__()
        assert w.groups in [
            1,
            w.in_channels,
        ], "Argument(g) only be 1 for conv2d, or be in_channels for depthwise conv2d"

        bias = True if w.bias is not None else False
        pad = True if (w.stride[0] == 1 and w.padding[0] == w.kernel_size[0] // 2) else False
        if w.groups == 1:
            conv = keras.layers.Conv2D(
                w.out_channels,
                w.kernel_size,
                w.stride,
                'SAME' if pad else 'VALID',
                dilation_rate=w.dilation,
                use_bias=bias,
                # torch[out, in, h, w] to TF[h, w, in, out]
                kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).detach().numpy()),
                bias_initializer=keras.initializers.Constant(w.bias.detach().numpy()) if bias else 'zeros',
            )
        else:
            conv = keras.layers.DepthwiseConv2D(
                w.kernel_size,
                w.stride,
                'SAME' if pad else 'VALID',
                dilation_rate=w.dilation,
                use_bias=bias,
                depthwise_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 0, 1).detach().numpy()),
                bias_initializer=keras.initializers.Constant(w.bias.detach().numpy()) if bias else 'zeros',
            )
        self.conv = conv if pad else keras.Sequential([TFPad(autopad(w.kernel_size[0], w.padding[0])), conv])

    def call(self, inputs):
        return self.conv(inputs)


class TFDense(keras.layers.Layer):
    def __init__(self, w=None):
        super().__init__()
        bias = False if w.bias is None else True
        self.fc = keras.layers.Dense(
            w.out_features,
            use_bias=True if bias else False,
            kernel_initializer=keras.initializers.Constant(w.weight.permute(1, 0).detach().numpy()),
            bias_initializer=keras.initializers.Constant(w.bias.detach().numpy()) if bias else 'zeros',
        )

    def call(self, inputs):
        return self.fc(inputs)


class TFBaseConv1d(keras.layers.Layer):
    """Standard convolution1d or depthwiseconv1d depends on 'g' argument."""

    def __init__(self, w=None):
        super().__init__()
        assert w.groups in [1, w.in_channels], "Argument(g) only be 1 for conv1d, or be inp for depthwise conv1d"

        bias = True if w.bias is not None else False
        pad = True if (w.stride[0] == 1 and w.padding[0] == w.kernel_size[0] // 2) else False
        if w.groups == 1:
            conv = keras.layers.Conv1D(
                w.out_channels,
                w.kernel_size,
                w.stride,
                'SAME' if pad else 'VALID',
                dilation_rate=w.dilation,
                use_bias=bias,
                kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 1, 0).detach().numpy()),
                bias_initializer=keras.initializers.Constant(w.bias.detach().numpy()) if bias else 'zeros',
            )
        else:
            conv = keras.layers.DepthwiseConv1D(
                w.kernel_size,
                w.stride,
                'SAME' if pad else 'VALID',
                dilation_rate=w.dilation,
                use_bias=bias,
                depthwise_initializer=keras.initializers.Constant(w.weight.permute(2, 0, 1).detach().numpy()),
                bias_initializer=keras.initializers.Constant(w.bias.detach().numpy()) if bias else 'zeros',
            )
        padding = keras.layers.ZeroPadding1D(padding=autopad(w.kernel_size[0], w.padding[0]))
        self.conv = conv if pad else keras.Sequential([padding, conv])

    def call(self, inputs):
        return self.conv(inputs)


class TFAADownsample(keras.layers.Layer):
    """DepthwiseConv1D with fixed weights only for audio model."""

    def __init__(self, w=None):
        super().__init__()
        pad = True if w.stride == 1 else False

        filt = keras.layers.DepthwiseConv1D(
            w.filt_size,
            w.stride,
            'SAME' if pad else 'VALID',
            use_bias=False,
            depthwise_initializer=keras.initializers.Constant(w.filt.permute(2, 0, 1).detach().numpy()),
        )
        padding = keras.layers.ZeroPadding1D(padding=autopad(w.filt_size, None))
        self.filt = filt if pad else keras.Sequential([padding, filt])

    def call(self, inputs):
        return self.filt(inputs)


class TFActivation(keras.layers.Layer):
    """Activation functions."""

    def __init__(self, w=None):
        super().__init__()

        if isinstance(w, nn.ReLU):
            act = keras.layers.ReLU()
        elif isinstance(w, nn.ReLU6):
            act = keras.layers.ReLU(max_value=6)
        elif isinstance(w, nn.LeakyReLU):
            act = keras.layers.LeakyReLU(w.negative_slope)
        elif isinstance(w, nn.Sigmoid):
            act = lambda x: keras.activations.sigmoid(x)  # noqa
        else:
            raise Exception(f'no matching TensorFlow activation found for PyTorch activation {w}')
        self.act = act

    def call(self, inputs):
        return self.act(inputs)


def tf_pool(w=None):
    """Pooling functions."""
    if isinstance(w, nn.MaxPool2d):
        return TFMaxPool2d(w)
    elif isinstance(w, nn.AvgPool2d):
        return TFAvgPool2d(w)
    elif isinstance(w, nn.AdaptiveAvgPool2d):
        return keras.layers.GlobalAveragePooling2D()
    elif isinstance(w, nn.AdaptiveAvgPool1d):
        return keras.layers.GlobalAveragePooling1D()
    else:
        raise Exception(f'no matching pool function found for {w}')


class TFUpsample(keras.layers.Layer):
    # TF version of torch.nn.Upsample()
    def __init__(self, w=None):
        super().__init__()
        scale_factor, mode = (
            (int(w.scale_factor), w.mode)
            if isinstance(w, nn.Upsample)
            else (int(w.kwargs["scale_factor"]), w.kwargs["mode"])
        )
        assert scale_factor % 2 == 0, "scale_factor must be multiple of 2"
        self.upsample = lambda x: tf.image.resize(x, (x.shape[1] * scale_factor, x.shape[2] * scale_factor), mode)
        # self.upsample = keras.layers.UpSampling2D(size=scale_factor, interpolation=mode)
        # with default arguments: align_corners=False, half_pixel_centers=False
        # self.upsample = lambda x: tf.raw_ops.ResizeNearestNeighbor(images=x,
        #                                                            size=(x.shape[1] * 2, x.shape[2] * 2))

    def call(self, inputs):
        return self.upsample(inputs)


def tf_method(node):
    """Tensorflow version for bulid-in method."""
    if 'size' in node.name:
        if len(node.args) == 2:
            # dim = -1 if node.args[1] == 1 else node.args[1]
            return eval(str(node.args[0])).shape[-1]
        else:
            n, h, w, c = eval(str(node.args[0])).shape.as_list()
            return [n, c, h, w]

    elif 'getitem' in node.name:
        return eval(str(node.args[0]))[node.args[1]]
    elif 'floordiv' in node.name:
        return eval(str(node.args[0])) // 2
    elif 'contiguous' in node.name:
        return eval(str(node.args[0]))
    elif 'mul' in node.name:
        return node.args[0] * eval(str(node.args[1]))
    else:
        raise Exception(f'No match method found for {node.name}')
