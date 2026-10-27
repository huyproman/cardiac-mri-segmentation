from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from common.labels import FOREGROUND_CLASSES

DICE_SMOOTH = 1e-6
HD95_SPACING = (1.25, 1.25)

_SLICE_STRUCTURE = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class ClassMetric:
    class_id: int
    dice: float
    hd95_mean: float
    hd95_std: float


def dice_score(
    prediction: np.ndarray, target: np.ndarray, smooth: float = DICE_SMOOTH
) -> float:
    p = np.asarray(prediction, dtype=bool)
    t = np.asarray(target, dtype=bool)
    intersection = float(np.count_nonzero(p & t))
    denominator = float(np.count_nonzero(p) + np.count_nonzero(t))
    return (2.0 * intersection + smooth) / (denominator + smooth)


def dice_per_class(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
    classes: Sequence[int] = FOREGROUND_CLASSES,
    smooth: float = DICE_SMOOTH,
) -> dict[int, float]:
    if prediction_labels.shape != target_labels.shape:
        raise ValueError(f"shape mismatch: {prediction_labels.shape} vs {target_labels.shape}")
    return {
        c: dice_score(prediction_labels == c, target_labels == c, smooth) for c in classes
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    return mask & ~binary_erosion(mask, structure=_SLICE_STRUCTURE, border_value=0)


def hd95_slice(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float] = HD95_SPACING,
) -> float:
    p = np.asarray(prediction, dtype=bool)
    t = np.asarray(target, dtype=bool)
    if not p.any() or not t.any():
        return float("nan")

    sp, st = _surface(p), _surface(t)
    if not sp.any() or not st.any():
        sp, st = p, t

    dist_to_t = distance_transform_edt(~st, sampling=spacing)
    dist_to_p = distance_transform_edt(~sp, sampling=spacing)
    distances = np.concatenate([dist_to_t[sp], dist_to_p[st]])
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def hd95_per_class(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
    classes: Sequence[int] = FOREGROUND_CLASSES,
    spacing: tuple[float, float] = HD95_SPACING,
) -> dict[int, tuple[float, float]]:
    if prediction_labels.ndim != 3:
        raise ValueError(f"expected a 3D label volume, got shape {prediction_labels.shape}")

    results: dict[int, tuple[float, float]] = {}
    for c in classes:
        values = [
            hd95_slice(prediction_labels[:, :, z] == c, target_labels[:, :, z] == c, spacing)
            for z in range(prediction_labels.shape[2])
        ]
        arr = np.asarray(values, dtype=float)
        if np.all(np.isnan(arr)):
            results[c] = (float("nan"), float("nan"))
        else:
            results[c] = (float(np.nanmean(arr)), float(np.nanstd(arr)))
    return results


def evaluate_volume(
    prediction_labels: np.ndarray,
    target_labels: np.ndarray,
    classes: Sequence[int] = FOREGROUND_CLASSES,
    smooth: float = DICE_SMOOTH,
    spacing: tuple[float, float] = HD95_SPACING,
) -> list[ClassMetric]:
    classes = tuple(classes)
    dice = dice_per_class(prediction_labels, target_labels, classes, smooth)
    hd95 = hd95_per_class(prediction_labels, target_labels, classes, tuple(spacing))
    return [ClassMetric(c, dice[c], hd95[c][0], hd95[c][1]) for c in classes]


def mean_dice(metrics: Sequence[ClassMetric]) -> float:
    return float(np.mean([m.dice for m in metrics]))


def mean_hd95(metrics: Sequence[ClassMetric]) -> float:
    values = [m.hd95_mean for m in metrics]
    return float(np.nanmean(values)) if values else float("nan")
