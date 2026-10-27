from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from common.labels import FOREGROUND_CLASSES

BACKGROUND_MODE = "fuse"
BACKGROUND_WEIGHT = 0.5

INITIAL_WEIGHT = 0.5
WEIGHT_BOUNDS = (0.0, 1.0)
MAX_ITERATIONS = 300
OPTIMIZER_TOL = 1e-6
DICE_EPSILON = 1e-6


@dataclass(frozen=True)
class FusionWeights:

    class_ids: tuple[int, ...]
    weights: tuple[float, ...]
    background_mode: str = BACKGROUND_MODE
    background_weight: float = BACKGROUND_WEIGHT

    def __post_init__(self) -> None:
        if len(self.class_ids) != len(self.weights):
            raise ValueError("class_ids and weights must have equal length")
        for w in self.weights:
            if not 0.0 <= w <= 1.0:
                raise ValueError(f"fusion weight {w} outside [0, 1]")

    def as_mapping(self) -> dict[int, float]:
        return dict(zip(self.class_ids, self.weights))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FusionWeights":
        data = json.loads(Path(path).read_text())
        return cls(
            class_ids=tuple(int(c) for c in data["class_ids"]),
            weights=tuple(float(w) for w in data["weights"]),
            background_mode=data.get("background_mode", BACKGROUND_MODE),
            background_weight=float(data.get("background_weight", BACKGROUND_WEIGHT)),
        )


def _validate(p2d: np.ndarray, p3d: np.ndarray) -> None:
    if p2d.shape != p3d.shape:
        raise ValueError(f"prediction shapes differ: {p2d.shape} vs {p3d.shape}")
    if p2d.ndim < 2:
        raise ValueError("predictions must have a trailing class axis")


def fuse_classwise_probabilities(
    p2d: np.ndarray,
    p3d: np.ndarray,
    weights: FusionWeights,
    renormalize: bool = False,
) -> np.ndarray:
    _validate(p2d, p3d)
    num_classes = p2d.shape[-1]
    mapping = weights.as_mapping()
    unknown = [c for c in mapping if not 0 <= c < num_classes]
    if unknown:
        raise ValueError(f"class ids {unknown} outside the prediction's {num_classes} channels")

    fused = np.empty_like(p2d, dtype=np.float32)
    for c in range(num_classes):
        if c in mapping:
            w = mapping[c]
        elif weights.background_mode == "zero":
            fused[..., c] = 0.0
            continue
        elif weights.background_mode == "from_2d":
            fused[..., c] = p2d[..., c]
            continue
        else:
            w = weights.background_weight
        fused[..., c] = w * p2d[..., c] + (1.0 - w) * p3d[..., c]

    if renormalize:
        total = fused.sum(axis=-1, keepdims=True)
        fused = np.divide(fused, total, out=np.zeros_like(fused), where=total > 0)
    return fused


def soft_dice(prediction: np.ndarray, target: np.ndarray, epsilon: float = DICE_EPSILON) -> float:
    numerator = 2.0 * float(np.sum(prediction * target))
    denominator = float(np.sum(prediction**2) + np.sum(target**2)) + epsilon
    return numerator / denominator


def classwise_dice_objective(
    weight_vector: Sequence[float],
    p2d: np.ndarray,
    p3d: np.ndarray,
    onehot_target: np.ndarray,
    class_ids: Sequence[int],
    epsilon: float = DICE_EPSILON,
) -> float:
    scores = []
    for w, c in zip(weight_vector, class_ids):
        w = float(np.clip(w, 0.0, 1.0))
        fused_c = w * p2d[..., c] + (1.0 - w) * p3d[..., c]
        scores.append(soft_dice(fused_c, onehot_target[..., c], epsilon))
    return float(np.mean(scores))


def optimize_classwise_weights(
    p2d_val: np.ndarray,
    p3d_val: np.ndarray,
    onehot_target_val: np.ndarray,
    foreground_classes: Sequence[int] = FOREGROUND_CLASSES,
    initial_weights: Sequence[float] | None = None,
    bounds: tuple[float, float] = WEIGHT_BOUNDS,
    max_iterations: int = MAX_ITERATIONS,
    tol: float = OPTIMIZER_TOL,
    background_mode: str = BACKGROUND_MODE,
    background_weight: float = BACKGROUND_WEIGHT,
) -> tuple[FusionWeights, dict]:
    _validate(p2d_val, p3d_val)
    if onehot_target_val.shape != p2d_val.shape:
        raise ValueError(
            f"target shape {onehot_target_val.shape} does not match predictions {p2d_val.shape}"
        )

    class_ids = tuple(int(c) for c in foreground_classes)
    if initial_weights is None:
        initial_weights = [INITIAL_WEIGHT] * len(class_ids)
    x0 = np.asarray(initial_weights, dtype=np.float64)
    if x0.shape[0] != len(class_ids):
        raise ValueError("initial_weights length must match foreground_classes")

    lo, hi = bounds

    def negative_objective(x: np.ndarray) -> float:
        clipped = np.clip(x, lo, hi)
        return -classwise_dice_objective(
            clipped, p2d_val, p3d_val, onehot_target_val, class_ids
        )

    result = minimize(
        negative_objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": max_iterations, "xatol": tol, "fatol": tol, "disp": False},
    )

    best = tuple(float(np.clip(v, lo, hi)) for v in np.atleast_1d(result.x))
    weights = FusionWeights(
        class_ids=class_ids,
        weights=best,
        background_mode=background_mode,
        background_weight=background_weight,
    )
    summary = {
        "objective": float(-result.fun),
        "iterations": int(result.nit),
        "function_evaluations": int(result.nfev),
        "success": bool(result.success),
        "message": str(result.message),
    }
    return weights, summary
