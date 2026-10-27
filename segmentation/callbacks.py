from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from segmentation.schedules import (
    INITIAL_LEARNING_RATE,
    MIN_LEARNING_RATE,
    CosineAnnealingWarmRestarts,
)

SEED = 42
EARLY_STOPPING_PATIENCE = 50
MONITOR = "val_loss"
CHECKPOINT_FILENAME = "best.weights"


def fold_checkpoint_path(output_dir: str | Path, fold_index: int) -> Path:
    path = Path(output_dir) / f"fold_{fold_index}" / CHECKPOINT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_exists(checkpoint_path: str | Path) -> bool:
    return Path(f"{checkpoint_path}.index").is_file()


def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_mixed_precision(enabled: bool = True) -> None:
    policy = "mixed_float16" if enabled else "float32"
    tf.keras.mixed_precision.set_global_policy(policy)


def build_callbacks(
    checkpoint_path: str | Path,
    first_restart_epoch: int,
    cycle_multiplier: float,
    initial_learning_rate: float = INITIAL_LEARNING_RATE,
    min_learning_rate: float = MIN_LEARNING_RATE,
    patience: int = EARLY_STOPPING_PATIENCE,
    monitor: str = MONITOR,
) -> list[tf.keras.callbacks.Callback]:
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        CosineAnnealingWarmRestarts(
            first_restart_epoch, cycle_multiplier, initial_learning_rate, min_learning_rate
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
