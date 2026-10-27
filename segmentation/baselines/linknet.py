from __future__ import annotations

import tensorflow as tf

from common.labels import NUM_CLASSES

from segmentation.blocks import conv_block_2d, conv_bn_relu_2d

layers = tf.keras.layers


def linknet(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    channels: int = 32,
    name: str = "LinkNet",
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)

    def encode(x, out_ch: int):
        y = conv_bn_relu_2d(x, out_ch, k=3, s=2)
        return conv_bn_relu_2d(y, out_ch, k=3)

    def decode(x, skip, out_ch: int):
        x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
        x = conv_bn_relu_2d(x, out_ch, k=1)
        x = layers.Add()([x, skip])
        return conv_bn_relu_2d(x, out_ch, k=3)

    e1 = conv_block_2d(inputs, channels)
    e2 = encode(e1, channels * 2)
    e3 = encode(e2, channels * 4)
    e4 = encode(e3, channels * 8)
    e5 = encode(e4, channels * 16)

    d4 = decode(e5, e4, channels * 8)
    d3 = decode(d4, e3, channels * 4)
    d2 = decode(d3, e2, channels * 2)
    d1 = decode(d2, e1, channels)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax", dtype="float32")(d1)
    return tf.keras.Model(inputs, outputs, name=name)
