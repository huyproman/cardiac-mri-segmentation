from __future__ import annotations

import tensorflow as tf

from common.labels import NUM_CLASSES

from segmentation.blocks import attention_gate_3d, conv_block_3d

layers = tf.keras.layers


def attention_unet_3d(
    input_shape: tuple[int, int, int, int] = (224, 224, 10, 1),
    num_classes: int = NUM_CLASSES,
    channels: int = 32,
    name: str = "AttentionUNet3D",
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)

    c1 = conv_block_3d(inputs, channels)
    p1 = layers.MaxPooling3D((2, 2, 1))(c1)
    c2 = conv_block_3d(p1, channels * 2)
    p2 = layers.MaxPooling3D((2, 2, 1))(c2)
    c3 = conv_block_3d(p2, channels * 4)
    p3 = layers.MaxPooling3D((2, 2, 1))(c3)
    c4 = conv_block_3d(p3, channels * 8)
    p4 = layers.MaxPooling3D((2, 2, 1))(c4)

    c5 = conv_block_3d(p4, channels * 16)

    g4 = attention_gate_3d(c4, c5, channels * 8)
    u6 = layers.Conv3DTranspose(channels * 8, (2, 2, 1), strides=(2, 2, 1), padding="same")(c5)
    c6 = conv_block_3d(layers.Concatenate()([u6, g4]), channels * 8)

    g3 = attention_gate_3d(c3, c6, channels * 4)
    u7 = layers.Conv3DTranspose(channels * 4, (2, 2, 1), strides=(2, 2, 1), padding="same")(c6)
    c7 = conv_block_3d(layers.Concatenate()([u7, g3]), channels * 4)

    g2 = attention_gate_3d(c2, c7, channels * 2)
    u8 = layers.Conv3DTranspose(channels * 2, (2, 2, 1), strides=(2, 2, 1), padding="same")(c7)
    c8 = conv_block_3d(layers.Concatenate()([u8, g2]), channels * 2)

    g1 = attention_gate_3d(c1, c8, channels)
    u9 = layers.Conv3DTranspose(channels, (2, 2, 1), strides=(2, 2, 1), padding="same")(c8)
    c9 = conv_block_3d(layers.Concatenate()([u9, g1]), channels)

    outputs = layers.Conv3D(num_classes, (1, 1, 1), activation="softmax", dtype="float32")(c9)
    return tf.keras.Model(inputs, outputs, name=name)
