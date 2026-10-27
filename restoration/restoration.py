from __future__ import annotations

from typing import Sequence

import numpy as np

from common.geometry import CropMetadata, restore_from_crop
from common.labels import BACKGROUND_LABEL


def restore_labels_to_full(
    roi_labels: np.ndarray,
    crops: Sequence[CropMetadata],
    background_label: int = BACKGROUND_LABEL,
) -> np.ndarray:
    if roi_labels.ndim != 3:
        raise ValueError(f"expected a 3D label volume, got shape {roi_labels.shape}")
    if roi_labels.shape[2] != len(crops):
        raise ValueError("number of slices must equal the number of crop metadata entries")

    height, width = crops[0].source_shape
    out = np.full((height, width, roi_labels.shape[2]), background_label, dtype=roi_labels.dtype)
    for z, meta in enumerate(crops):
        out[:, :, z] = restore_from_crop(roi_labels[:, :, z], meta, background_label)
    return out


def scatter_slices_to_original_depth(
    restored: np.ndarray,
    slice_indices: Sequence[int],
    original_depth: int,
    background_label: int = BACKGROUND_LABEL,
) -> np.ndarray:
    if restored.shape[2] != len(slice_indices):
        raise ValueError("slice_indices length must match the restored volume depth")
    out = np.full(
        (restored.shape[0], restored.shape[1], original_depth),
        background_label,
        dtype=restored.dtype,
    )
    for position, z in enumerate(slice_indices):
        if not 0 <= z < original_depth:
            raise IndexError(f"slice index {z} outside original depth {original_depth}")
        out[:, :, z] = restored[:, :, position]
    return out
