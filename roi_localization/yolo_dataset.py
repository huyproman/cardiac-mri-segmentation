from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml

from common.geometry import bbox_from_mask
from common.nifti import VolumeRecord

HEART_CLASS_ID = 0


def bbox_to_yolo_line(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    class_id: int = HEART_CLASS_ID,
) -> str:
    x_min, y_min, x_max, y_max = bbox
    h, w = shape
    row_center = (x_min + x_max) / 2.0 / h
    col_center = (y_min + y_max) / 2.0 / w
    box_h = (x_max - x_min) / h
    box_w = (y_max - y_min) / w
    return f"{class_id} {col_center:.6f} {row_center:.6f} {box_w:.6f} {box_h:.6f}"


def slice_to_png_array(slice_2d: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(slice_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(finite.min()), float(finite.max())
    scaled = np.zeros_like(finite) if hi <= lo else (finite - lo) / (hi - lo)
    return (scaled * 255.0).astype(np.uint8)


def export_records(records: Iterable[VolumeRecord], output_dir: str | Path) -> int:
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for record in records:
        if record.mask is None:
            continue
        for z in range(record.num_slices):
            bbox = bbox_from_mask(record.mask[:, :, z])
            if bbox is None:
                continue
            stem = f"{record.patient_id}_{record.phase}_slice{z:03d}"
            Image.fromarray(slice_to_png_array(record.image[:, :, z])).save(
                output_dir / f"{stem}.png"
            )
            line = bbox_to_yolo_line(bbox, record.image.shape[:2])
            (output_dir / f"{stem}.txt").write_text(line + "\n")
            written += 1
    return written


def write_data_yaml(
    root: str | Path,
    train_dir: str,
    val_dir: str,
    test_dir: str | None = None,
    class_names: Sequence[str] = ("heart",),
) -> Path:
    root = Path(root)
    payload: dict = {
        "train": str(root / train_dir),
        "val": str(root / val_dir),
        "nc": len(class_names),
        "names": list(class_names),
    }
    if test_dir is not None:
        payload["test"] = str(root / test_dir)
    path = root / "data.yaml"
    root.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path
