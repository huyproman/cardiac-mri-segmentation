from __future__ import annotations

import tensorflow as tf

from common.labels import NUM_CLASSES

from segmentation.blocks import conv_block_2d, residual_conv_block_2d, up_concat_2d

layers = tf.keras.layers


def unet(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    channels: int = 32,
    name: str = "UNet",
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)

    c1 = conv_block_2d(inputs, channels)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = conv_block_2d(p1, channels * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = conv_block_2d(p2, channels * 4)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = conv_block_2d(p3, channels * 8)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    c5 = conv_block_2d(p4, channels * 16)

    c6 = conv_block_2d(up_concat_2d(c5, c4, channels * 8), channels * 8)
    c7 = conv_block_2d(up_concat_2d(c6, c3, channels * 4), channels * 4)
    c8 = conv_block_2d(up_concat_2d(c7, c2, channels * 2), channels * 2)
    c9 = conv_block_2d(up_concat_2d(c8, c1, channels), channels)

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax", dtype="float32")(c9)
    return tf.keras.Model(inputs, outputs, name=name)


def resunet(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    channels: int = 32,
    name: str = "ResUNet",
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)

    c1 = residual_conv_block_2d(inputs, channels)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = residual_conv_block_2d(p1, channels * 2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = residual_conv_block_2d(p2, channels * 4)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = residual_conv_block_2d(p3, channels * 8)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    c5 = residual_conv_block_2d(p4, channels * 16)

    c6 = residual_conv_block_2d(up_concat_2d(c5, c4, channels * 8), channels * 8)
    c7 = residual_conv_block_2d(up_concat_2d(c6, c3, channels * 4), channels * 4)
    c8 = residual_conv_block_2d(up_concat_2d(c7, c2, channels * 2), channels * 2)
    c9 = residual_conv_block_2d(up_concat_2d(c8, c1, channels), channels)

    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax", dtype="float32")(c9)
    return tf.keras.Model(inputs, outputs, name=name)
