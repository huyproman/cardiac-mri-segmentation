from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterator, Sequence

import numpy as np

N_SPLITS = 5
SPLIT_SEED = 42


@dataclass(frozen=True)
class Fold:
    index: int
    train_patients: tuple[str, ...]
    val_patients: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train_patients) & set(self.val_patients)
        if overlap:
            raise ValueError(f"fold {self.index} has patient overlap: {sorted(overlap)}")


def patient_folds(
    patient_ids: Sequence[str],
    n_splits: int = N_SPLITS,
    shuffle: bool = True,
    seed: int = SPLIT_SEED,
) -> list[Fold]:
    unique = sorted(set(patient_ids))
    if len(unique) < n_splits:
        raise ValueError(f"cannot build {n_splits} folds from {len(unique)} patients")

    order = np.arange(len(unique))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)

    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    for position, idx in enumerate(order):
        buckets[position % n_splits].append(unique[idx])

    folds: list[Fold] = []
    for i in range(n_splits):
        val = sorted(buckets[i])
        train = sorted(p for j, b in enumerate(buckets) if j != i for p in b)
        folds.append(Fold(index=i, train_patients=tuple(train), val_patients=tuple(val)))
    return folds


def split_records_by_patient(records: Sequence, fold: Fold) -> tuple[list, list]:
    train_set = set(fold.train_patients)
    val_set = set(fold.val_patients)
    train = [r for r in records if r.patient_id in train_set]
    val = [r for r in records if r.patient_id in val_set]
    return train, val


def group_indices_by_patient(patient_ids: Sequence[Hashable]) -> dict[Hashable, list[int]]:
    groups: dict[Hashable, list[int]] = {}
    for i, pid in enumerate(patient_ids):
        groups.setdefault(pid, []).append(i)
    return groups


def iter_folds(
    records: Sequence,
    n_splits: int = N_SPLITS,
    shuffle: bool = True,
    seed: int = SPLIT_SEED,
) -> Iterator[tuple[Fold, list, list]]:
    ids = [r.patient_id for r in records]
    for fold in patient_folds(ids, n_splits, shuffle, seed):
        train, val = split_records_by_patient(records, fold)
        yield fold, train, val
