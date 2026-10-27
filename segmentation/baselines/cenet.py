from __future__ import annotations

import tensorflow as tf

from common.labels import NUM_CLASSES

layers = tf.keras.layers


def _bn_relu(x):
    return layers.ReLU()(layers.BatchNormalization()(x))


def _conv3x3(x, c: int, s: int = 1, d: int = 1):
    return layers.Conv2D(c, 3, strides=s, dilation_rate=d, padding="same", use_bias=False)(x)


def _conv1x1(x, c: int, s: int = 1):
    return layers.Conv2D(c, 1, strides=s, padding="same", use_bias=False)(x)


def _res_basic_block(x, c: int, s: int = 1):
    shortcut = x
    y = _bn_relu(_conv3x3(x, c, s=s))
    y = layers.BatchNormalization()(_conv3x3(y, c, s=1))
    if s != 1 or shortcut.shape[-1] != c:
        shortcut = layers.BatchNormalization()(_conv1x1(shortcut, c, s=s))
    return layers.ReLU()(layers.Add()([y, shortcut]))


def _make_layer(x, c: int, n_blocks: int, stride_first: int):
    x = _res_basic_block(x, c, s=stride_first)
    for _ in range(1, n_blocks):
        x = _res_basic_block(x, c, s=1)
    return x


def _resnet34_encoder(inp):
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inp)
    x = _bn_relu(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    c1 = _make_layer(x, 64, n_blocks=3, stride_first=1)
    c2 = _make_layer(c1, 128, n_blocks=4, stride_first=2)
    c3 = _make_layer(c2, 256, n_blocks=6, stride_first=2)
    c4 = _make_layer(c3, 512, n_blocks=3, stride_first=2)
    return c1, c2, c3, c4


def _dac_block(x, c: int):
    b1 = _bn_relu(_conv3x3(x, c, d=1))
    b2 = _bn_relu(_conv1x1(_bn_relu(_conv3x3(x, c, d=3)), c))
    b3 = _bn_relu(_conv1x1(_bn_relu(_conv3x3(x, c, d=5)), c))
    b4 = _bn_relu(_conv1x1(_bn_relu(_conv3x3(_bn_relu(_conv1x1(x, c)), c, d=1)), c))
    y = _bn_relu(_conv1x1(layers.Concatenate()([b1, b2, b3, b4]), c))
    return _bn_relu(layers.Add()([y, x]))


class ResidualMultiKernelPooling(layers.Layer):

    def __init__(self, channels: int, bins: tuple[int, ...] = (1, 2, 3, 6), **kwargs) -> None:
        super().__init__(**kwargs)
        self.channels = channels
        self.bins = tuple(bins)

    def build(self, input_shape) -> None:
        self.proj = layers.Conv2D(self.channels, 1, padding="same", use_bias=False)
        self.bn = layers.BatchNormalization()
        self.act = layers.ReLU()
        super().build(input_shape)

    def call(self, x):
        height, width = tf.shape(x)[1], tf.shape(x)[2]
        pooled = [x]
        for b in self.bins:
            y = tf.image.resize(x, (b, b), method="bilinear")
            pooled.append(tf.image.resize(y, (height, width), method="bilinear"))
        y = tf.concat(pooled, axis=-1)
        return self.act(self.bn(self.proj(y)))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"channels": self.channels, "bins": self.bins})
        return config


def _decoder_block(x, skip, c: int):
    x = _bn_relu(_conv1x1(x, c))
    x = _bn_relu(layers.Conv2DTranspose(c, 3, strides=2, padding="same", use_bias=False)(x))
    x = _bn_relu(_conv1x1(x, c))
    x = layers.Concatenate()([x, skip])
    return _bn_relu(_conv3x3(x, c))


def _upsample_to_input(x, out_ch: int):
    for _ in range(2):
        x = layers.Conv2DTranspose(out_ch, 3, strides=2, padding="same", use_bias=False)(x)
        x = _bn_relu(x)
    return x


def cenet(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    name: str = "CENet",
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)
    x = layers.Concatenate()([inputs] * 3) if input_shape[-1] == 1 else inputs

    c1, c2, c3, c4 = _resnet34_encoder(x)
    b = _bn_relu(_conv1x1(c4, 512))
    b = _dac_block(b, 512)
    b = ResidualMultiKernelPooling(512, bins=(1, 2, 3, 6), name="rmp")(b)

    d3 = _decoder_block(b, c3, 256)
    d2 = _decoder_block(d3, c2, 128)
    d1 = _decoder_block(d2, c1, 64)
    u0 = _upsample_to_input(d1, 64)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax", dtype="float32")(u0)
    return tf.keras.Model(inputs, outputs, name=name)
