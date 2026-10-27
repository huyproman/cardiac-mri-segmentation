from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import nibabel as nib
import numpy as np

PHASES = ("ED", "ES")


@dataclass
class VolumeRecord:

    patient_id: str
    phase: str
    image: np.ndarray
    spacing: tuple[float, float, float]
    affine: np.ndarray
    mask: np.ndarray | None = None
    slice_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.image.ndim != 3:
            raise ValueError(f"image must be 3D, got shape {self.image.shape}")
        if self.mask is not None and self.mask.shape != self.image.shape:
            raise ValueError(
                f"mask shape {self.mask.shape} does not match image shape {self.image.shape}"
            )
        if self.slice_indices is None:
            object.__setattr__(self, "slice_indices", tuple(range(self.image.shape[2])))
        elif len(self.slice_indices) != self.image.shape[2]:
            raise ValueError("slice_indices length does not match image depth")

    @property
    def num_slices(self) -> int:
        return self.image.shape[2]

    @property
    def key(self) -> tuple[str, str]:
        return (self.patient_id, self.phase)


def load_nifti(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float], np.ndarray]:
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, zooms, np.asarray(img.affine, dtype=np.float64)


def save_nifti(path: str | Path, array: np.ndarray, affine: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.asarray(affine, dtype=np.float64)), str(path))


def load_volume(
    image_path: str | Path,
    mask_path: str | Path | None,
    patient_id: str,
    phase: str,
) -> VolumeRecord:
    image, spacing, affine = load_nifti(image_path)
    mask = None
    if mask_path is not None:
        mask_array, _, _ = load_nifti(mask_path)
        mask = np.rint(mask_array).astype(np.uint8)
    return VolumeRecord(
        patient_id=patient_id,
        phase=phase,
        image=image,
        spacing=spacing,
        affine=affine,
        mask=mask,
    )


def iter_acdc_processed(
    root: str | Path,
    split: str,
    require_mask: bool = True,
) -> Iterator[VolumeRecord]:
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split directory not found: {split_dir}")
    for patient_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        patient_id = patient_dir.name
        for phase in PHASES:
            image_path = patient_dir / f"{patient_id}_{phase}_processed.nii.gz"
            mask_path = patient_dir / f"{patient_id}_{phase}_gt_processed.nii.gz"
            if not image_path.exists():
                continue
            if require_mask and not mask_path.exists():
                continue
            yield load_volume(
                image_path,
                mask_path if mask_path.exists() else None,
                patient_id,
                phase,
            )


def patient_id_from_name(name: str) -> str:
    stem = Path(name).name
    if stem.startswith("patient"):
        return stem[: len("patient") + 3]
    raise ValueError(f"cannot derive patient id from '{name}'")
