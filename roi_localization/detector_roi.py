from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from common.geometry import (
    ROI_CROP_SIZE,
    ROI_PAD_VALUE,
    CropMetadata,
    bbox_center,
    crop_with_context,
)
from roi_localization.detection import (
    DETECTION_CONF_THRESHOLD,
    Detection,
    select_detection,
)
from roi_localization.detector_pruning import (
    MAX_INTERIOR_GAP,
    MIN_BOX_AREA_FRACTION,
    select_inference_slices,
)


@dataclass
class DetectorRoiVolume:
    patient_id: str
    phase: str
    image: np.ndarray
    crops: list[CropMetadata]
    slice_indices: tuple[int, ...]
    centers: list[tuple[int, int]]

    @property
    def num_slices(self) -> int:
        return self.image.shape[2]


def centers_from_detections(
    detections_per_slice: Sequence[Sequence[Detection]],
    image_shape: tuple[int, int],
    min_confidence: float = DETECTION_CONF_THRESHOLD,
) -> list[tuple[int, int]]:
    raw: list[tuple[int, int] | None] = []
    for dets in detections_per_slice:
        chosen = select_detection(dets, min_confidence)
        raw.append(bbox_center(chosen.bbox) if chosen is not None else None)

    default = (image_shape[0] // 2, image_shape[1] // 2)
    resolved: list[tuple[int, int]] = []
    for i, center in enumerate(raw):
        if center is not None:
            resolved.append(center)
            continue
        replacement = default
        for offset in range(1, len(raw)):
            found = False
            for probe in (i - offset, i + offset):
                if 0 <= probe < len(raw) and raw[probe] is not None:
                    replacement = raw[probe]
                    found = True
                    break
            if found:
                break
        resolved.append(replacement)
    return resolved


def extract_roi_from_detections(
    volume: np.ndarray,
    detections_per_slice: Sequence[Sequence[Detection]],
    patient_id: str,
    phase: str,
    crop_size: tuple[int, int] = ROI_CROP_SIZE,
    pad_value: float = ROI_PAD_VALUE,
    slice_indices: Sequence[int] | None = None,
    prune_slices: bool = False,
    min_confidence: float = DETECTION_CONF_THRESHOLD,
    min_box_area_fraction: float = MIN_BOX_AREA_FRACTION,
    max_interior_gap: int = MAX_INTERIOR_GAP,
) -> DetectorRoiVolume:
    if volume.shape[2] != len(detections_per_slice):
        raise ValueError("detections must be provided for every slice")

    shape = (volume.shape[0], volume.shape[1])
    source_indices = tuple(slice_indices or range(volume.shape[2]))

    if prune_slices:
        kept = select_inference_slices(
            detections_per_slice,
            shape,
            min_confidence,
            min_box_area_fraction,
            max_interior_gap,
        )
    else:
        kept = tuple(range(volume.shape[2]))
    if not kept:
        kept = tuple(range(volume.shape[2]))

    kept_detections = [detections_per_slice[z] for z in kept]
    centers = centers_from_detections(kept_detections, shape, min_confidence)

    crop_size = tuple(crop_size)
    out = np.empty((*crop_size, len(kept)), dtype=np.float32)
    metas: list[CropMetadata] = []
    for position, z in enumerate(kept):
        crop, meta = crop_with_context(volume[:, :, z], centers[position], crop_size, pad_value)
        out[:, :, position] = crop
        metas.append(meta)

    return DetectorRoiVolume(
        patient_id=patient_id,
        phase=phase,
        image=out,
        crops=metas,
        slice_indices=tuple(source_indices[z] for z in kept),
        centers=centers,
    )
