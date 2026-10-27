from __future__ import annotations

import tensorflow as tf

from common.labels import NUM_CLASSES

from segmentation.blocks import conv_block_2d, up_concat_2d

layers = tf.keras.layers


def _mlp(x, hidden_units: int, drop_rate: float = 0.0):
    y = layers.Dense(hidden_units, activation="gelu")(x)
    y = layers.Dropout(drop_rate)(y)
    y = layers.Dense(x.shape[-1])(y)
    return layers.Dropout(drop_rate)(y)


def _transformer_block(x, num_heads: int = 8, mlp_dim: int = 2048, drop_rate: float = 0.0):
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=int(x.shape[-1]) // num_heads, dropout=drop_rate
    )(y, y)
    x = layers.Add()([x, y])
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    return layers.Add()([x, _mlp(y, mlp_dim, drop_rate)])


def transunet(
    input_shape: tuple[int, int, int] = (224, 224, 1),
    num_classes: int = NUM_CLASSES,
    channels: int = 32,
    vit_dim: int = 512,
    vit_depth: int = 8,
    vit_heads: int = 8,
    vit_mlp: int = 2048,
    drop_rate: float = 0.0,
    name: str = "TransUNet",
) -> tf.keras.Model:
    inputs = layers.Input(input_shape)

    c1 = conv_block_2d(inputs, channels)
    c2 = conv_block_2d(layers.MaxPooling2D(2)(c1), channels * 2)
    c3 = conv_block_2d(layers.MaxPooling2D(2)(c2), channels * 4)
    c4 = conv_block_2d(layers.MaxPooling2D(2)(c3), channels * 8)
    b = conv_block_2d(layers.MaxPooling2D(2)(c4), channels * 16)

    h = layers.Conv2D(vit_dim, 1, padding="same")(b)
    h = layers.Add()([h, layers.Conv2D(vit_dim, 3, padding="same")(h)])

    grid_h, grid_w = h.shape[1], h.shape[2]
    if grid_h is None or grid_w is None:
        raise ValueError("TransUNet requires a statically known input resolution")

    t = layers.Reshape((grid_h * grid_w, vit_dim))(h)
    for _ in range(vit_depth):
        t = _transformer_block(t, num_heads=vit_heads, mlp_dim=vit_mlp, drop_rate=drop_rate)
    z = layers.Reshape((grid_h, grid_w, vit_dim))(t)
    z = layers.Conv2D(channels * 16, 1, padding="same")(z)

    d4 = conv_block_2d(up_concat_2d(z, c4, channels * 8), channels * 8)
    d3 = conv_block_2d(up_concat_2d(d4, c3, channels * 4), channels * 4)
    d2 = conv_block_2d(up_concat_2d(d3, c2, channels * 2), channels * 2)
    d1 = conv_block_2d(up_concat_2d(d2, c1, channels), channels)

    outputs = layers.Conv2D(num_classes, 1, activation="softmax", dtype="float32")(d1)
    return tf.keras.Model(inputs, outputs, name=name)
