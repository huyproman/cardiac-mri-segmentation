from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.ndimage import binary_closing, generate_binary_structure, label

from common.labels import FOREGROUND_CLASSES

ENTROPY_THRESHOLD = 0.9
ENTROPY_EPSILON = 1e-10
LCCF_PROBABILITY_THRESHOLD = 0.37
LCCF_CONNECTIVITY = 3
CLOSING_STRUCTURE_SIZE = 3
CLOSING_ITERATIONS = 1


@dataclass
class RefinementTrace:

    entropy: np.ndarray
    entropy_filtered: np.ndarray
    component_masks: dict[int, np.ndarray]
    closed_masks: dict[int, np.ndarray]


def normalize_class_scores(scores: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(scores, dtype=np.float32), 0.0, None)
    total = p.sum(axis=-1, keepdims=True)
    uniform = np.float32(1.0 / p.shape[-1])
    return np.where(total > 0.0, np.divide(p, total, out=np.zeros_like(p), where=total > 0.0), uniform)


def predictive_entropy(probabilities: np.ndarray, epsilon: float = ENTROPY_EPSILON) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float32)
    return -np.sum(p * np.log(p + epsilon), axis=-1)


def apply_entropy_threshold(
    probabilities: np.ndarray,
    threshold: float = ENTROPY_THRESHOLD,
    foreground_classes: Sequence[int] = FOREGROUND_CLASSES,
    epsilon: float = ENTROPY_EPSILON,
    renormalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    if renormalize:
        p = normalize_class_scores(probabilities)
    else:
        p = np.array(probabilities, dtype=np.float32, copy=True)
    entropy = predictive_entropy(p, epsilon)
    uncertain = entropy >= threshold
    for c in foreground_classes:
        p[..., c] = np.where(uncertain, 0.0, p[..., c])
    return p, entropy


def largest_connected_component(
    binary: np.ndarray, connectivity: int = LCCF_CONNECTIVITY
) -> np.ndarray:
    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool)
    structure = generate_binary_structure(binary.ndim, min(connectivity, binary.ndim))
    labelled, count = label(binary, structure=structure)
    if count <= 1:
        return binary.astype(bool)
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return labelled == int(sizes.argmax())


def morphological_closing_per_slice(
    mask: np.ndarray,
    structure_size: int = CLOSING_STRUCTURE_SIZE,
    iterations: int = CLOSING_ITERATIONS,
) -> np.ndarray:
    structure = np.ones((structure_size, structure_size), dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    for z in range(mask.shape[2]):
        out[:, :, z] = binary_closing(
            mask[:, :, z].astype(bool), structure=structure, iterations=iterations
        )
    return out


def refine_prediction(
    fused_probabilities: np.ndarray,
    foreground_classes: Sequence[int] = FOREGROUND_CLASSES,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    entropy_epsilon: float = ENTROPY_EPSILON,
    lccf_probability_threshold: float = LCCF_PROBABILITY_THRESHOLD,
    lccf_connectivity: int = LCCF_CONNECTIVITY,
    closing_structure_size: int = CLOSING_STRUCTURE_SIZE,
    closing_iterations: int = CLOSING_ITERATIONS,
    return_trace: bool = False,
    renormalize: bool = True,
) -> np.ndarray | tuple[np.ndarray, RefinementTrace]:
    p = np.asarray(fused_probabilities, dtype=np.float32)
    if p.ndim != 4:
        raise ValueError(f"expected a 4D (x, y, z, C) tensor, got shape {p.shape}")

    filtered, entropy = apply_entropy_threshold(
        p, entropy_threshold, foreground_classes, entropy_epsilon, renormalize
    )

    component_masks: dict[int, np.ndarray] = {}
    closed_masks: dict[int, np.ndarray] = {}
    for c in foreground_classes:
        binary = filtered[..., c] > lccf_probability_threshold
        component = largest_connected_component(binary, lccf_connectivity)
        component_masks[c] = component
        closed_masks[c] = morphological_closing_per_slice(
            component, closing_structure_size, closing_iterations
        )

    labels = np.zeros(p.shape[:3], dtype=np.uint8)
    best_score = np.zeros(p.shape[:3], dtype=np.float32)
    for c in foreground_classes:
        mask = closed_masks[c]
        score = np.where(mask, np.maximum(filtered[..., c], 1e-6), 0.0)
        take = mask & (score > best_score)
        labels[take] = c
        best_score[take] = score[take]

    if return_trace:
        trace = RefinementTrace(
            entropy=entropy,
            entropy_filtered=filtered,
            component_masks=component_masks,
            closed_masks=closed_masks,
        )
        return labels, trace
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-2d", required=True)
    parser.add_argument("--pred-3d", required=True)
    parser.add_argument("--weights", required=True, help="JSON produced by optimize_weights.py")
    parser.add_argument("--output", required=True, help=".npy destination for the label volume")
    parser.add_argument("--save-fused", help="optional .npy destination for fused probabilities")
    parser.add_argument("--entropy-threshold", type=float, default=ENTROPY_THRESHOLD)
    parser.add_argument("--lccf-threshold", type=float, default=LCCF_PROBABILITY_THRESHOLD)
    parser.add_argument("--lccf-connectivity", type=int, default=LCCF_CONNECTIVITY)
    parser.add_argument(
        "--renormalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="renormalize fused scores over the class axis before entropy (default: on)",
    )
    return parser.parse_args()


def main() -> None:
    from fusion.classwise import FusionWeights, fuse_classwise_probabilities

    args = parse_args()
    p2d = np.load(args.pred_2d)
    p3d = np.load(args.pred_3d)
    weights = FusionWeights.load(args.weights)

    fused = fuse_classwise_probabilities(p2d, p3d, weights, args.renormalize)
    if args.save_fused:
        Path(args.save_fused).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_fused, fused)

    labels = refine_prediction(
        fused,
        foreground_classes=weights.class_ids,
        entropy_threshold=args.entropy_threshold,
        lccf_probability_threshold=args.lccf_threshold,
        lccf_connectivity=args.lccf_connectivity,
        renormalize=args.renormalize,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, labels)

    present, counts = np.unique(labels, return_counts=True)
    print("label histogram:", dict(zip(present.tolist(), counts.tolist())))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
