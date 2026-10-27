from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from common.labels import FOREGROUND_CLASSES
from common.nifti import VolumeRecord

MIN_CLASS_AREA = 30
MAX_CONSECUTIVE_PARTIAL = 2


@dataclass(frozen=True)
class SliceClassification:
    background_only: tuple[int, ...]
    full: tuple[int, ...]
    partial: tuple[int, ...]


def class_areas(mask_slice: np.ndarray, classes: Sequence[int]) -> dict[int, int]:
    return {c: int(np.count_nonzero(mask_slice == c)) for c in classes}


def classify_slices(
    mask: np.ndarray,
    min_class_area: int = MIN_CLASS_AREA,
    required_classes: Sequence[int] = FOREGROUND_CLASSES,
) -> SliceClassification:
    background_only: list[int] = []
    full: list[int] = []
    partial: list[int] = []
    for z in range(mask.shape[2]):
        plane = mask[:, :, z]
        if not np.any(plane > 0):
            background_only.append(z)
            continue
        areas = class_areas(plane, required_classes)
        if all(area >= min_class_area for area in areas.values()):
            full.append(z)
        else:
            partial.append(z)
    return SliceClassification(tuple(background_only), tuple(full), tuple(partial))


def select_slices(
    mask: np.ndarray,
    min_class_area: int = MIN_CLASS_AREA,
    required_classes: Sequence[int] = FOREGROUND_CLASSES,
    max_consecutive_partial: int = MAX_CONSECUTIVE_PARTIAL,
    drop_background_only: bool = True,
) -> tuple[int, ...]:
    classification = classify_slices(mask, min_class_area, required_classes)

    if not classification.full:
        return ()

    background_only = set(classification.background_only)
    lo, hi = classification.full[0], classification.full[-1]

    keep = {
        z for z in range(lo, hi + 1) if z not in background_only or not drop_background_only
    }

    for step, bound in ((-1, -1), (1, mask.shape[2])):
        anchor = lo if step < 0 else hi
        retained = 0
        z = anchor + step
        while z != bound and retained < max_consecutive_partial:
            if z in background_only and drop_background_only:
                break
            keep.add(z)
            retained += 1
            z += step

    return tuple(sorted(keep))


def prune_volume(
    record: VolumeRecord,
    min_class_area: int = MIN_CLASS_AREA,
    required_classes: Sequence[int] = FOREGROUND_CLASSES,
    max_consecutive_partial: int = MAX_CONSECUTIVE_PARTIAL,
    drop_background_only: bool = True,
) -> VolumeRecord:
    if record.mask is None:
        raise ValueError(
            "mask-based pruning requires an annotation mask; "
            "use roi_localization.detector_pruning for inference-time pruning"
        )
    kept = select_slices(
        record.mask,
        min_class_area,
        required_classes,
        max_consecutive_partial,
        drop_background_only,
    )
    index = list(kept)
    source_indices = record.slice_indices or tuple(range(record.num_slices))
    return VolumeRecord(
        patient_id=record.patient_id,
        phase=record.phase,
        image=record.image[:, :, index],
        mask=record.mask[:, :, index],
        spacing=record.spacing,
        affine=record.affine,
        slice_indices=tuple(source_indices[i] for i in index),
    )
