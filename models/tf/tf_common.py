import tensorflow as tf
from tensorflow import keras
import numpy as np


class TFBN(keras.layers.Layer):
    def __init__(self, w=None, name=None):
        super().__init__()
        self.n = [f'{name}.bias',
                  f'{name}.weight',
                  f'{name}.running_mean',
                  f'{name}.running_var']
        self.bn = keras.layers.BatchNormalization(
            beta_initializer=keras.initializers.Constant(w[self.n[0]].numpy()),
            gamma_initializer=keras.initializers.Constant(w[self.n[1]].numpy()),
            moving_mean_initializer=keras.initializers.Constant(w[self.n[2]].numpy()),
            moving_variance_initializer=keras.initializers.Constant(w[self.n[3]].numpy()),
            epsilon=1e-5,
            momentum=0.1)

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


class TFBaseConv2d(keras.layers.Layer):
    # Standard convolution2d or depthwiseconv2d depends on 'g' argument.
    def __init__(self, inp, oup, kernel, stride, padding, g=1, bias=True,
                 act=None, bn=True, names=None, w=None):
        super().__init__()
        assert g in [1, inp], "Argument(g) only be 1 for conv2d, or be inp for depthwise conv2d"
        self.n = names if isinstance(names, list) else [names]
        self.n[0] = [f'{self.n[0]}.weight', f'{self.n[0]}.bias']

        if g == 1:
            conv = keras.layers.Conv2D(
                oup,
                kernel,
                stride,
                'SAME' if stride == 1 and padding == kernel // 2 else 'VALID',
                use_bias=bias,
                kernel_initializer=keras.initializers.Constant(w[self.n[0][0]].permute(2, 3, 1, 0).numpy()),
                bias_initializer=keras.initializers.Constant(w[self.n[0][1]].numpy()) if bias else 'zeros'
            )
        else:
            conv = keras.layers.DepthwiseConv2D(
                kernel,
                stride,
                'SAME' if stride == 1 and padding == kernel // 2 else 'VALID',
                use_bias=bias,
                depthwise_initializer=keras.initializers.Constant(w[self.n[0][0]].permute(2, 3, 0, 1).numpy()),
                bias_initializer=keras.initializers.Constant(w[self.n[0][1]].numpy()) if bias else 'zeros'
            )
        self.conv = conv if stride == 1 and padding == kernel // 2 else \
            keras.Sequential([TFPad(autopad(kernel, padding)), conv])
        self.bn = TFBN(w=w, name=self.n[1]) if bn else tf.identity

        if act == "silu":
            self.act = lambda x: keras.activations.swish(x)
        elif act == "relu":
            self.act = lambda x: keras.activations.relu(x)
        elif act == "lrelu":
            self.act = lambda x: keras.activations.relu(x, alpha=0.1)
        elif act is None:
            self.act = None
        else:
            raise AttributeError("Unsupported act type: {}".format(act))

    def call(self, inputs):
        return self.act(self.bn(self.conv(inputs))) if self.act else self.bn(self.conv(inputs))


class TFInvertedResidual(keras.layers.Layer):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6, name=None, w=None):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect
        self.n = [f'{name}.conv.{i}' for i in range(0, 8) if (i != 2 and i != 5)]

        self.conv = keras.Sequential([
            TFBaseConv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False,
                         act='relu', bn=True, names=self.n[:2], w=w),
            TFBaseConv2d(inp * expand_ratio,
                         inp * expand_ratio,
                         3,
                         stride,
                         1,
                         g=inp * expand_ratio,
                         bias=False,
                         act='relu',
                         bn=True,
                         names=self.n[2:4],
                         w=w),
            TFBaseConv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False,
                         bn=True, names=self.n[4:], w=w)])

    def call(self, inputs):
        if self.use_res_connect:
            return inputs + self.conv(inputs)
        else:
            return self.conv(inputs)


class TFDense(keras.layers.Layer):
    def __init__(self, oup, bias=True, name=None, w=None):
        super().__init__()
        self.n = [f'{name}.weight',
                  f'{name}.bias']

        self.fc = keras.layers.Dense(oup,
                                     use_bias=True if bias else False,
                                     kernel_initializer=keras.initializers.Constant(w[self.n[0]].permute(1, 0).numpy()),
                                     bias_initializer=keras.initializers.Constant(w[self.n[1]].numpy()) if bias else 'zeros',
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


class PFLDInference(keras.layers.Layer):
    def __init__(self, names=None, w=None):
        super().__init__()

        self.conv1 = TFBaseConv2d(3, 32, kernel=3, stride=2, padding=1, bias=False,
                                  act='relu', bn=True, names=names[0], w=w)
        self.conv2 = TFBaseConv2d(32, 32, kernel=3, stride=1, padding=1, bias=False,
                                  act='relu', bn=True, names=names[1], w=w)

        self.conv3_1 = TFInvertedResidual(32, 16, 2, False, 2, name=names[2], w=w)

        self.block3_2 = TFInvertedResidual(16, 16, 1, True, 2, name=names[3], w=w)
        self.block3_3 = TFInvertedResidual(16, 16, 1, True, 2, name=names[4], w=w)
        self.block3_4 = TFInvertedResidual(16, 16, 1, True, 2, name=names[5], w=w)
        self.block3_5 = TFInvertedResidual(16, 16, 1, True, 2, name=names[6], w=w)

        self.conv4_1 = TFInvertedResidual(16, 32, 2, False, 2, name=names[7], w=w)

        self.conv5_1 = TFInvertedResidual(32, 32, 1, False, 4, name=names[8], w=w)
        self.block5_2 = TFInvertedResidual(32, 32, 1, True, 4, name=names[9], w=w)
        self.block5_3 = TFInvertedResidual(32, 32, 1, True, 4, name=names[10], w=w)
        self.block5_4 = TFInvertedResidual(32, 32, 1, True, 4, name=names[11], w=w)
        self.block5_5 = TFInvertedResidual(32, 32, 1, True, 4, name=names[12], w=w)
        self.block5_6 = TFInvertedResidual(32, 32, 1, True, 4, name=names[13], w=w)
        #
        self.conv6_1 = TFInvertedResidual(32, 16, 1, False, 2, name=names[14], w=w)  # [16, 14, 14]

        self.conv7 = TFBaseConv2d(16, 32, kernel=3, stride=2, padding=1, bias=False,
                                  act='relu', bn=True, names=names[15], w=w)  # [32, 7, 7]
        self.conv8 = TFBaseConv2d(32, 32, kernel=7, stride=1, padding=0, bias=True,
                                  act='relu', bn=False, names=names[16], w=w)  # [128, 1, 1]

        self.avg_pool1 = keras.layers.GlobalAveragePooling2D()
        self.fc = TFDense(2, bias=True, name=names[17], w=w)

    def call(self, inputs):  # x: 112, 112, 3
        x = self.conv1(inputs)  # [64, 56, 56]
        x = self.conv2(x)  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)

        x = self.conv7(x)
        x2 = self.avg_pool1(x)

        x3 = self.avg_pool1(self.conv8(x))

        multi_scale = tf.concat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks


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