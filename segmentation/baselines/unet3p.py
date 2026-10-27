from __future__ import annotations

import tensorflow as tf

from common.labels import NUM_CLASSES

from segmentation.blocks import conv_block_2d, resize_like

layers = tf.keras.layers


def unet3p(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    channels: int = 32,
    name: str = "UNet3Plus",
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)

    c1 = conv_block_2d(inputs, channels)
    c2 = conv_block_2d(layers.MaxPooling2D(2)(c1), channels * 2)
    c3 = conv_block_2d(layers.MaxPooling2D(2)(c2), channels * 4)
    c4 = conv_block_2d(layers.MaxPooling2D(2)(c3), channels * 8)
    c5 = conv_block_2d(layers.MaxPooling2D(2)(c4), channels * 16)

    d4 = conv_block_2d(
        layers.Concatenate()(
            [resize_like(c1, c4), resize_like(c2, c4), resize_like(c3, c4), c4, resize_like(c5, c4)]
        ),
        channels * 8,
    )
    d3 = conv_block_2d(
        layers.Concatenate()(
            [
                resize_like(c1, c3),
                resize_like(c2, c3),
                c3,
                resize_like(c4, c3),
                resize_like(c5, c3),
                resize_like(d4, c3),
            ]
        ),
        channels * 4,
    )
    d2 = conv_block_2d(
        layers.Concatenate()(
            [
                resize_like(c1, c2),
                c2,
                resize_like(c3, c2),
                resize_like(c4, c2),
                resize_like(c5, c2),
                resize_like(d4, c2),
                resize_like(d3, c2),
            ]
        ),
        channels * 2,
    )
    d1 = conv_block_2d(
        layers.Concatenate()(
            [
                c1,
                resize_like(c2, c1),
                resize_like(c3, c1),
                resize_like(c4, c1),
                resize_like(c5, c1),
                resize_like(d4, c1),
                resize_like(d3, c1),
                resize_like(d2, c1),
            ]
        ),
        channels,
    )

    outputs = layers.Conv2D(num_classes, 1, activation="softmax", dtype="float32")(d1)
    return tf.keras.Model(inputs, outputs, name=name)
