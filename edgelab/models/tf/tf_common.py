import tensorflow as tf
from tensorflow import keras
import numpy as np


class TFBN(keras.layers.Layer):
    def __init__(self, w=None):
        super().__init__()
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w.bias.detach().numpy()),
            gamma_initializer=keras.initializers.Constant(w.weight.detach().numpy()),
            moving_mean_initializer=keras.initializers.Constant(w.running_mean.detach().numpy()),
            moving_variance_initializer=keras.initializers.Constant(w.running_var.detach().numpy()),
            epsilon=w.eps,
            momentum=w.momentum)

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
        self.pad = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])

    def call(self, inputs):
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
        assert w.groups in [1, w.in_channels], "Argument(g) only be 1 for conv2d, or be inp for depthwise conv2d"

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
                kernel_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 1, 0).detach().numpy()),
                bias_initializer=keras.initializers.Constant(w.bias.detach().numpy()) if bias else 'zeros'
            )
        else:
            conv = keras.layers.DepthwiseConv2D(
                w.kernel_size,
                w.stride,
                'SAME' if pad else 'VALID',
                dilation_rate=w.dilation,
                use_bias=bias,
                depthwise_initializer=keras.initializers.Constant(w.weight.permute(2, 3, 0, 1).detach().numpy()),
                bias_initializer=keras.initializers.Constant(w.bias.detach().numpy()) if bias else 'zeros'
            )
        self.conv = conv if pad else keras.Sequential([TFPad(autopad(w.kernel_size[0], w.padding[0])), conv])

    def call(self, inputs):
        return self.conv(inputs)


class TFDense(keras.layers.Layer):
    def __init__(self, w=None):
        super().__init__()
        bias = False if w.bias is None else True
        self.fc = keras.layers.Dense(w.out_features,
                                     use_bias=True if bias else False,
                                     kernel_initializer=keras.initializers.Constant(
                                         w.weight.permute(1, 0).detach().numpy()),
                                     bias_initializer=keras.initializers.Constant(
                                         w.bias.detach().numpy()) if bias else 'zeros',
                                     )

    def call(self, inputs):
        return self.fc(inputs)


class TFBaseConv1d(keras.layers.Layer):
    # Standard convolution1d or depthwiseconv1d depends on 'g' argument.
    def __init__(self, inp, oup, kernel, stride, padding, g=1, bias=False, dilation=1,
                 act=None, bn=False, name=None, w=None):
        super(TFBaseConv1d, self).__init__()
        assert g in [1, inp], "Argument(g) only be 1 for conv1d, or be inp for depthwise conv1d"
        assert not (dilation != 1 and stride != 1), "dilation_rate value != 1 is " \
                                                    "incompatible with specifying any stride value != 1!"
        self.n = [[f'{name}.0.weight', f'{name}.0.bias'], f'{name}.1'] if bn \
            else [[f'{name}.weight', f'{name}.bias']]

        if g == 1:
            conv = keras.layers.Conv1D(
                oup,
                kernel,
                stride,
                'SAME' if stride == 1 and padding == kernel // 2 else 'VALID',
                use_bias=bias,
                kernel_initializer=keras.initializers.Constant(w[self.n[0][0]].permute(2, 1, 0).numpy()),
                bias_initializer=keras.initializers.Constant(w[self.n[0][1]].numpy()) if bias else 'zeros'
            )
        else:
            conv = keras.layers.DepthwiseConv1D(
                kernel,
                stride,
                'SAME' if stride == 1 and padding == kernel // 2 else 'VALID',
                dilation_rate=dilation,
                use_bias=bias,
                depthwise_initializer=keras.initializers.Constant(w[self.n[0][0]].permute(2, 0, 1).numpy()),
                bias_initializer=keras.initializers.Constant(w[self.n[0][1]].numpy()) if bias else 'zeros'
            )
        pad = keras.layers.ZeroPadding1D(padding=autopad(kernel, padding))
        self.conv = conv if stride == 1 and padding == kernel // 2 else keras.Sequential([pad, conv])
        self.bn = TFBN(w=w, name=self.n[1]) if bn else tf.identity

        if act == "silu":
            self.act = lambda x: keras.activations.swish(x)
        elif act == "relu":
            self.act = lambda x: keras.activations.relu(x)
        elif act == "lrelu":
            self.act = lambda x: keras.activations.relu(x, alpha=0.2)
        elif act is None:
            self.act = None
        else:
            raise AttributeError("Unsupported act type: {}".format(act))

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs))) if self.act else self.bn(self.conv(inputs))


class TFAADownsample(keras.layers.Layer):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super().__init__()
        self.filt_size = filt_size
        self.stride = stride

        ha = np.arange(1, filt_size // 2 + 2, 1)
        a = np.concatenate([ha, np.flip(ha, axis=-1)[1:]], axis=-1).astype(float)
        a = np.tile(a[None, None, :] / np.sum(a, axis=-1), (channels, 1, 1))
        a = a.transpose((2, 0, 1))

        filt = keras.layers.DepthwiseConv1D(
            a.shape[0],
            stride,
            'SAME' if stride == 1 else 'VALID',
            use_bias=False,
            depthwise_initializer=keras.initializers.Constant(a)
        )
        pad = keras.layers.ZeroPadding1D(padding=autopad(filt_size, None))
        self.filt = filt if stride == 1 else keras.Sequential([pad, filt])

    def call(self, inputs):
        return self.filt(inputs)


class TFDown(keras.layers.Layer):
    def __init__(self, channels, d=2, k=3, name=None, w=None):
        super().__init__()
        kk = d + 1
        self.n = f'{name}.down'

        self.down = keras.Sequential([
            TFBaseConv1d(channels,
                         channels * 2,
                         kernel=kk,
                         stride=1,
                         padding=kk // 2,
                         bias=False,
                         act='lrelu',
                         bn=True,
                         name=self.n,
                         w=w),
            TFAADownsample(channels=channels * 2, stride=d, filt_size=k)
        ])

    def call(self, inputs):
        return self.down(inputs)


class TFResBlock1d(keras.layers.Layer):
    def __init__(self, dim, dilation=1, kernel_size=3, name=None, w=None):
        super().__init__()
        self.n = [f'{name}.block_t', f'{name}.block_f', f'{name}.shortcut']
        self.block_t = TFBaseConv1d(
            dim,
            dim,
            kernel=kernel_size,
            stride=1,
            padding=dilation * (kernel_size // 2),
            g=dim,
            dilation=dilation,
            bias=False,
            act='lrelu',
            bn=True,
            name=self.n[0],
            w=w,
        )
        self.block_f = TFBaseConv1d(dim, dim, 1, 1, 0, bias=False, act='lrelu', bn=True, name=self.n[1], w=w)
        self.shortcut = TFBaseConv1d(dim, dim, 1, 1, 0, bias=True, name=self.n[2], w=w)

    def call(self, inputs):
        return self.shortcut(inputs) + self.block_f(inputs) + self.block_t(inputs)


class Audio_Backbone(keras.layers.Layer):
    def __init__(self, nf=2, clip_length=None, factors=[4, 4, 4], out_channel=32, name='backbone', w=None):
        super().__init__()
        base_ = 4
        self.start = TFBaseConv1d(1, nf, 11, 6, 5, bias=False, act='lrelu', bn=True, name=f'{name}.start', w=w)

        model = []
        n = 0
        for i, f in enumerate(factors):
            model.append(TFDown(channels=nf, d=f, k=f * 2 + 1, name=f'{name}.down.{n}', w=w))
            nf *= 2
            if i % 2 == 0:
                n += 1
                model.append(TFResBlock1d(dim=nf, dilation=1, kernel_size=7, name=f'{name}.down.{n}', w=w))
            n += 1
        self.down = keras.Sequential(model)

        factors = [2, 2]
        model = []
        n = 0
        for _, f in enumerate(factors):
            for i in range(1):
                for j in range(3):
                    model.append(TFResBlock1d(dim=nf, dilation=3**j, kernel_size=7, name=f'{name}.down2.{n}', w=w))
                    n += 1
            model.append(TFDown(channels=nf, d=f, k=f * 2 + 1, name=f'{name}.down2.{n}', w=w))
            n += 1
            nf *= 2
        self.down2 = keras.Sequential(model)
        self.project = TFBaseConv1d(nf, out_channel, 1, 1, 0, bias=True, name=f'{name}.project', w=w)
        self.clip_length = clip_length

    def call(self, inputs):
        x = self.start(inputs)
        x = self.down(x)
        x = self.down2(x)
        feature = self.project(x)

        return feature


class Audio_head(keras.layers.Layer):
    def __init__(self, in_channels, n_classes, drop=0.5, name='cls_head', w=None):
        super().__init__()
        self.avg = keras.layers.GlobalAveragePooling1D()
        self.fc = TFDense(in_channels, bias=True, name=f'{name}.fc', w=w)
        self.fc1 = TFDense(n_classes, bias=True, name=f'{name}.fc1', w=w)
        self.softmax = keras.layers.Softmax(axis=-1)
        # # self.dp = keras.layers.Dropout(rate=0.5, training=True)

    def call(self, inputs):
        return self.softmax(self.fc1(self.fc(self.avg(inputs))))