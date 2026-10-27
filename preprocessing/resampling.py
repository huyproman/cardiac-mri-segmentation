from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.ndimage import zoom

from common.labels import NUM_CLASSES
from common.nifti import VolumeRecord

TARGET_IN_PLANE_SPACING = (1.25, 1.25)
IMAGE_INTERPOLATION_ORDER = 1
MASK_INTERPOLATION_ORDER = 0
NORMALIZATION_EPSILON = 1e-10


def resample_in_plane(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    target_in_plane: tuple[float, float] = TARGET_IN_PLANE_SPACING,
    order: int = IMAGE_INTERPOLATION_ORDER,
) -> np.ndarray:
    factors = (spacing[0] / target_in_plane[0], spacing[1] / target_in_plane[1], 1.0)
    if np.allclose(factors, 1.0):
        return volume.astype(volume.dtype, copy=True)
    return zoom(volume, zoom=factors, order=order, mode="nearest")


def resampled_affine(
    affine: np.ndarray,
    spacing: tuple[float, float, float],
    target_in_plane: tuple[float, float] = TARGET_IN_PLANE_SPACING,
) -> np.ndarray:
    new_affine = np.array(affine, dtype=np.float64, copy=True)
    for axis, target in enumerate(target_in_plane):
        current = spacing[axis]
        if current > 0:
            new_affine[:3, axis] *= target / current
    return new_affine


def zscore_normalize(
    volume: np.ndarray,
    over_nonzero: bool = True,
    epsilon: float = NORMALIZATION_EPSILON,
) -> np.ndarray:
    values = volume[volume != 0] if over_nonzero else volume
    if values.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    mean = float(values.mean())
    std = float(values.std())
    return ((volume - mean) / (std + epsilon)).astype(np.float32)


def clean_mask_labels(mask: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    rounded = np.rint(mask).astype(np.int16)
    return np.clip(rounded, 0, num_classes - 1).astype(np.uint8)


def normalize_record(
    record: VolumeRecord,
    over_nonzero: bool = True,
    epsilon: float = NORMALIZATION_EPSILON,
) -> VolumeRecord:
    return replace(
        record,
        image=zscore_normalize(record.image, over_nonzero=over_nonzero, epsilon=epsilon),
    )


def preprocess_volume(
    record: VolumeRecord,
    target_in_plane: tuple[float, float] = TARGET_IN_PLANE_SPACING,
    image_order: int = IMAGE_INTERPOLATION_ORDER,
    mask_order: int = MASK_INTERPOLATION_ORDER,
    normalize_over_nonzero: bool = True,
    normalization_epsilon: float = NORMALIZATION_EPSILON,
    normalize: bool = True,
) -> VolumeRecord:
    target = tuple(target_in_plane)

    image = resample_in_plane(record.image, record.spacing, target, order=image_order)
    mask = None
    if record.mask is not None:
        mask = resample_in_plane(
            record.mask.astype(np.float32), record.spacing, target, order=mask_order
        )
        mask = clean_mask_labels(mask)

    if normalize:
        image = zscore_normalize(
            image, over_nonzero=normalize_over_nonzero, epsilon=normalization_epsilon
        )
    else:
        image = image.astype(np.float32, copy=False)

    new_spacing = (target[0], target[1], record.spacing[2])
    new_affine = resampled_affine(record.affine, record.spacing, target)

    return VolumeRecord(
        patient_id=record.patient_id,
        phase=record.phase,
        image=image,
        mask=mask,
        spacing=new_spacing,
        affine=new_affine,
        slice_indices=record.slice_indices,
    )


PAD_SLICE = -1


def depth_transform_indices(depth: int, target_depth: int) -> tuple[int, ...]:
    if depth == target_depth:
        return tuple(range(depth))
    if depth < target_depth:
        total = target_depth - depth
        before = total // 2
        after = total - before
        return (PAD_SLICE,) * before + tuple(range(depth)) + (PAD_SLICE,) * after
    start = (depth - target_depth) // 2
    return tuple(range(start, start + target_depth))


def crop_or_pad_depth(
    volume: np.ndarray,
    target_depth: int,
    pad_value: float = 0.0,
) -> np.ndarray:
    depth = volume.shape[2]
    if depth == target_depth:
        return volume
    if depth < target_depth:
        total = target_depth - depth
        before = total // 2
        after = total - before
        pad_width = [(0, 0), (0, 0), (before, after)]
        if volume.ndim == 4:
            pad_width.append((0, 0))
        return np.pad(volume, pad_width, mode="constant", constant_values=pad_value)
    start = (depth - target_depth) // 2
    return volume[:, :, start : start + target_depth]
