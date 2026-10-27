from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

BBox = tuple[int, int, int, int]

ROI_CROP_SIZE = (224, 224)
ROI_PAD_VALUE = 0.0


@dataclass(frozen=True)
class CropMetadata:

    source_shape: tuple[int, int]
    crop_size: tuple[int, int]
    center: tuple[int, int]
    source_bounds: tuple[int, int, int, int]
    pad_before: tuple[int, int]

    @property
    def valid_shape(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.source_bounds
        return (x1 - x0, y1 - y0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CropMetadata":
        return cls(
            source_shape=tuple(data["source_shape"]),
            crop_size=tuple(data["crop_size"]),
            center=tuple(data["center"]),
            source_bounds=tuple(data["source_bounds"]),
            pad_before=tuple(data["pad_before"]),
        )


def bbox_from_mask(mask_slice: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask_slice > 0)
    if coords.size == 0:
        return None
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    return (int(x_min), int(y_min), int(x_max) + 1, int(y_max) + 1)


def bbox_center(bbox: BBox) -> tuple[int, int]:
    x_min, y_min, x_max, y_max = bbox
    return ((x_min + x_max) // 2, (y_min + y_max) // 2)


def clip_bbox(bbox: BBox, shape: tuple[int, int]) -> BBox:
    x_min, y_min, x_max, y_max = bbox
    h, w = shape
    x_min = int(np.clip(x_min, 0, h))
    y_min = int(np.clip(y_min, 0, w))
    x_max = int(np.clip(x_max, 0, h))
    y_max = int(np.clip(y_max, 0, w))
    return (min(x_min, x_max), min(y_min, y_max), max(x_min, x_max), max(y_min, y_max))


def bbox_area(bbox: BBox) -> int:
    x_min, y_min, x_max, y_max = bbox
    return max(0, x_max - x_min) * max(0, y_max - y_min)


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def xyxy_to_xywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, x1 - x0, y1 - y0)


def denormalize_bbox(
    box: tuple[float, float, float, float], shape: tuple[int, int]
) -> tuple[float, float, float, float]:
    h, w = shape
    x0, y0, x1, y1 = box
    return (x0 * w, y0 * h, x1 * w, y1 * h)


def crop_with_context(
    array: np.ndarray,
    center: tuple[int, int],
    crop_size: tuple[int, int] = ROI_CROP_SIZE,
    pad_value: float = ROI_PAD_VALUE,
) -> tuple[np.ndarray, CropMetadata]:
    if array.ndim != 2:
        raise ValueError(f"expected a 2D array, got shape {array.shape}")
    h, w = array.shape
    crop_h, crop_w = crop_size
    cx, cy = center

    x0 = cx - crop_h // 2
    y0 = cy - crop_w // 2
    x1, y1 = x0 + crop_h, y0 + crop_w

    src_x0, src_x1 = max(x0, 0), min(x1, h)
    src_y0, src_y1 = max(y0, 0), min(y1, w)

    out = np.full(crop_size, pad_value, dtype=array.dtype)
    pad_x, pad_y = src_x0 - x0, src_y0 - y0
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[pad_x : pad_x + (src_x1 - src_x0), pad_y : pad_y + (src_y1 - src_y0)] = array[
            src_x0:src_x1, src_y0:src_y1
        ]

    meta = CropMetadata(
        source_shape=(h, w),
        crop_size=(crop_h, crop_w),
        center=(int(cx), int(cy)),
        source_bounds=(src_x0, src_y0, src_x1, src_y1),
        pad_before=(pad_x, pad_y),
    )
    return out, meta


def restore_from_crop(
    crop: np.ndarray,
    meta: CropMetadata,
    fill_value: float = 0.0,
    dtype: np.dtype | None = None,
) -> np.ndarray:
    if tuple(crop.shape) != tuple(meta.crop_size):
        raise ValueError(f"crop shape {crop.shape} does not match metadata {meta.crop_size}")
    out = np.full(meta.source_shape, fill_value, dtype=dtype or crop.dtype)
    x0, y0, x1, y1 = meta.source_bounds
    pad_x, pad_y = meta.pad_before
    if x1 > x0 and y1 > y0:
        out[x0:x1, y0:y1] = crop[pad_x : pad_x + (x1 - x0), pad_y : pad_y + (y1 - y0)]
    return out


def crop_volume_from_centers(
    volume: np.ndarray,
    centers: list[tuple[int, int]],
    crop_size: tuple[int, int] = ROI_CROP_SIZE,
    pad_value: float = ROI_PAD_VALUE,
) -> tuple[np.ndarray, list[CropMetadata]]:
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    if len(centers) != volume.shape[2]:
        raise ValueError("number of centers must equal the number of slices")
    crops = np.empty((*crop_size, volume.shape[2]), dtype=volume.dtype)
    metas: list[CropMetadata] = []
    for z, center in enumerate(centers):
        crop, meta = crop_with_context(volume[:, :, z], center, crop_size, pad_value)
        crops[:, :, z] = crop
        metas.append(meta)
    return crops, metas


def restore_volume_from_crops(
    crops: np.ndarray,
    metas: list[CropMetadata],
    fill_value: float = 0.0,
) -> np.ndarray:
    if crops.shape[2] != len(metas):
        raise ValueError("number of slices must equal the number of metadata entries")
    h, w = metas[0].source_shape
    out = np.full((h, w, crops.shape[2]), fill_value, dtype=crops.dtype)
    for z, meta in enumerate(metas):
        out[:, :, z] = restore_from_crop(crops[:, :, z], meta, fill_value)
    return out
