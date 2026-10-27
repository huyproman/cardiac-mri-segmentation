from __future__ import annotations

import tensorflow as tf

from common.labels import NUM_CLASSES

from segmentation.blocks import conv_block_2d

layers = tf.keras.layers


def _up(x, filters: int):
    return layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding="same")(x)


def unetpp(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    channels: int = 32,
    name: str = "UNetPlusPlus",
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)

    x00 = conv_block_2d(inputs, channels)
    x10 = conv_block_2d(layers.MaxPooling2D((2, 2))(x00), channels * 2)
    x20 = conv_block_2d(layers.MaxPooling2D((2, 2))(x10), channels * 4)
    x30 = conv_block_2d(layers.MaxPooling2D((2, 2))(x20), channels * 8)
    x40 = conv_block_2d(layers.MaxPooling2D((2, 2))(x30), channels * 16)

    x01 = conv_block_2d(layers.Concatenate()([_up(x10, channels), x00]), channels)
    x11 = conv_block_2d(layers.Concatenate()([_up(x20, channels * 2), x10]), channels * 2)
    x02 = conv_block_2d(layers.Concatenate()([_up(x11, channels), x00, x01]), channels)
    x21 = conv_block_2d(layers.Concatenate()([_up(x30, channels * 4), x20]), channels * 4)
    x12 = conv_block_2d(layers.Concatenate()([_up(x21, channels * 2), x10, x11]), channels * 2)
    x03 = conv_block_2d(layers.Concatenate()([_up(x12, channels), x00, x01, x02]), channels)
    x31 = conv_block_2d(layers.Concatenate()([_up(x40, channels * 8), x30]), channels * 8)
    x22 = conv_block_2d(layers.Concatenate()([_up(x31, channels * 4), x20, x21]), channels * 4)
    x13 = conv_block_2d(layers.Concatenate()([_up(x22, channels * 2), x10, x11, x12]), channels * 2)
    x04 = conv_block_2d(
        layers.Concatenate()([_up(x13, channels), x00, x01, x02, x03]), channels
    )

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax", dtype="float32")(x04)
    return tf.keras.Model(inputs, outputs, name=name)
