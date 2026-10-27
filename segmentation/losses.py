from __future__ import annotations

import tensorflow as tf

from common.labels import NUM_CLASSES

DICE_WEIGHT = 0.6
TVERSKY_WEIGHT = 0.4
TVERSKY_ALPHA = 0.7
TVERSKY_BETA = 0.3
LOSS_SMOOTH = 1e-6

K = tf.keras.backend


def _prepare(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
    return y_true, y_pred


def dice_loss(y_true, y_pred, num_classes: int = NUM_CLASSES, smooth: float = LOSS_SMOOTH):
    y_true, y_pred = _prepare(y_true, y_pred)
    scores = []
    for c in range(1, num_classes):
        t = K.flatten(y_true[..., c])
        p = K.flatten(y_pred[..., c])
        intersection = K.sum(t * p)
        scores.append((2.0 * intersection + smooth) / (K.sum(t) + K.sum(p) + smooth))
    return 1.0 - K.mean(K.stack(scores), axis=0)


def tversky_loss(
    y_true,
    y_pred,
    num_classes: int = NUM_CLASSES,
    alpha: float = TVERSKY_ALPHA,
    beta: float = TVERSKY_BETA,
    smooth: float = LOSS_SMOOTH,
):
    y_true, y_pred = _prepare(y_true, y_pred)
    scores = []
    for c in range(1, num_classes):
        t = K.flatten(y_true[..., c])
        p = K.flatten(y_pred[..., c])
        true_pos = K.sum(t * p)
        false_neg = K.sum(t * (1.0 - p))
        false_pos = K.sum((1.0 - t) * p)
        scores.append(
            (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
        )
    return 1.0 - K.mean(K.stack(scores), axis=0)


def make_hybrid_loss(
    num_classes: int = NUM_CLASSES,
    dice_weight: float = DICE_WEIGHT,
    tversky_weight: float = TVERSKY_WEIGHT,
    tversky_alpha: float = TVERSKY_ALPHA,
    tversky_beta: float = TVERSKY_BETA,
    smooth: float = LOSS_SMOOTH,
):

    def hybrid_dice_tversky(y_true, y_pred):
        d = dice_loss(y_true, y_pred, num_classes, smooth)
        t = tversky_loss(y_true, y_pred, num_classes, tversky_alpha, tversky_beta, smooth)
        return dice_weight * d + tversky_weight * t

    return hybrid_dice_tversky


def dice_metric(num_classes: int = NUM_CLASSES, smooth: float = LOSS_SMOOTH):
    def mean_foreground_dice(y_true, y_pred):
        return 1.0 - dice_loss(y_true, y_pred, num_classes, smooth)

    return mean_foreground_dice
