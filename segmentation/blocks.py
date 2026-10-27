from __future__ import annotations

import tensorflow as tf

layers = tf.keras.layers


def conv_block_2d(x, num_filters: int):
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def conv_block_3d(x, num_filters: int):
    x = layers.Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def attention_gate_2d(x, g, inter_channels: int):
    g = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(g)
    theta_x = layers.Conv2D(inter_channels, (1, 1), padding="same")(x)
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding="same")(g)
    act = layers.Activation("relu")(layers.Add()([theta_x, phi_g]))
    psi = layers.Conv2D(1, (1, 1), padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([x, psi])


def attention_gate_3d(x, g, inter_channels: int):
    g = layers.UpSampling3D(size=(2, 2, 1))(g)
    theta_x = layers.Conv3D(inter_channels, (1, 1, 1), padding="same")(x)
    phi_g = layers.Conv3D(inter_channels, (1, 1, 1), padding="same")(g)
    act = layers.Activation("relu")(layers.Add()([theta_x, phi_g]))
    psi = layers.Conv3D(1, (1, 1, 1), padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([x, psi])


def residual_conv_block_2d(x, num_filters: int):
    shortcut = layers.Conv2D(num_filters, (1, 1), padding="same")(x)
    y = layers.Conv2D(num_filters, (3, 3), padding="same")(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(num_filters, (3, 3), padding="same")(y)
    y = layers.BatchNormalization()(y)
    y = layers.Add()([y, shortcut])
    return layers.ReLU()(y)


def up_concat_2d(x, skip, num_filters: int):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding="same")(x)
    return layers.Concatenate()([x, skip])


def conv_bn_relu_2d(x, filters: int, k: int = 3, s: int = 1, d: int = 1):
    x = layers.Conv2D(filters, k, strides=s, padding="same", dilation_rate=d, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def resize_like(x, ref, method: str = "bilinear"):
    return layers.Lambda(
        lambda t: tf.image.resize(t[0], tf.shape(t[1])[1:3], method=method)
    )([x, ref])
