from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.geometry import (
    ROI_CROP_SIZE,
    ROI_PAD_VALUE,
    CropMetadata,
    bbox_center,
    bbox_from_mask,
    crop_with_context,
)
from common.nifti import VolumeRecord


@dataclass
class RoiVolume:

    patient_id: str
    phase: str
    image: np.ndarray
    crops: list[CropMetadata]
    mask: np.ndarray | None = None
    slice_indices: tuple[int, ...] | None = None

    @property
    def num_slices(self) -> int:
        return self.image.shape[2]


def slice_centers_from_mask(mask: np.ndarray) -> list[tuple[int, int] | None]:
    centers: list[tuple[int, int] | None] = []
    for z in range(mask.shape[2]):
        bbox = bbox_from_mask(mask[:, :, z])
        centers.append(bbox_center(bbox) if bbox is not None else None)
    return centers


def _fallback_center(
    centers: list[tuple[int, int] | None], index: int, shape: tuple[int, int]
) -> tuple[int, int]:
    for offset in range(1, len(centers)):
        for probe in (index - offset, index + offset):
            if 0 <= probe < len(centers) and centers[probe] is not None:
                return centers[probe]
    return (shape[0] // 2, shape[1] // 2)


def extract_roi_from_groundtruth(
    record: VolumeRecord,
    crop_size: tuple[int, int] = ROI_CROP_SIZE,
    pad_value: float = ROI_PAD_VALUE,
) -> RoiVolume:
    if record.mask is None:
        raise ValueError("ground-truth ROI extraction requires an annotation mask")
    crop_size = tuple(crop_size)

    centers = slice_centers_from_mask(record.mask)
    shape = record.image.shape[:2]

    image_out = np.empty((*crop_size, record.num_slices), dtype=np.float32)
    mask_out = np.empty((*crop_size, record.num_slices), dtype=record.mask.dtype)
    metas: list[CropMetadata] = []

    for z in range(record.num_slices):
        center = centers[z] or _fallback_center(centers, z, shape)
        img_crop, meta = crop_with_context(record.image[:, :, z], center, crop_size, pad_value)
        msk_crop, _ = crop_with_context(record.mask[:, :, z], center, crop_size, 0)
        image_out[:, :, z] = img_crop
        mask_out[:, :, z] = msk_crop
        metas.append(meta)

    return RoiVolume(
        patient_id=record.patient_id,
        phase=record.phase,
        image=image_out,
        mask=mask_out,
        crops=metas,
        slice_indices=record.slice_indices,
    )
