from __future__ import annotations

import tensorflow as tf

from common.labels import NUM_CLASSES

layers = tf.keras.layers


def _preact_res_block(x, num_filters: int, dropout_rate: float = 0.0):
    shortcut = layers.Conv2D(num_filters, 1, padding="same")(x)
    y = layers.BatchNormalization()(x)
    y = layers.ReLU()(y)
    y = layers.Conv2D(num_filters, 3, padding="same")(y)
    if dropout_rate > 0:
        y = layers.SpatialDropout2D(dropout_rate)(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)
    y = layers.Conv2D(num_filters, 3, padding="same")(y)
    return layers.Add()([y, shortcut])


def _upsample_conv(x, num_filters: int):
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    x = layers.Conv2D(num_filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def unet_v2(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    channels: int = 32,
    dropout: float = 0.0,
    name: str = "UNetV2",
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)

    e1 = _preact_res_block(inputs, channels, dropout)
    e2 = _preact_res_block(layers.MaxPooling2D(2)(e1), channels * 2, dropout)
    e3 = _preact_res_block(layers.MaxPooling2D(2)(e2), channels * 4, dropout)
    e4 = _preact_res_block(layers.MaxPooling2D(2)(e3), channels * 8, dropout)
    b = _preact_res_block(layers.MaxPooling2D(2)(e4), channels * 16, dropout)

    d4 = _preact_res_block(
        layers.Concatenate()([_upsample_conv(b, channels * 8), e4]), channels * 8, dropout
    )
    d3 = _preact_res_block(
        layers.Concatenate()([_upsample_conv(d4, channels * 4), e3]), channels * 4, dropout
    )
    d2 = _preact_res_block(
        layers.Concatenate()([_upsample_conv(d3, channels * 2), e2]), channels * 2, dropout
    )
    d1 = _preact_res_block(
        layers.Concatenate()([_upsample_conv(d2, channels), e1]), channels, dropout
    )

    outputs = layers.Conv2D(num_classes, 1, activation="softmax", dtype="float32")(d1)
    return tf.keras.Model(inputs, outputs, name=name)
