from __future__ import annotations

import math

import tensorflow as tf

INITIAL_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-4


def cosine_annealing_lr(
    epoch_in_cycle: int,
    cycle_length: float,
    initial_lr: float = INITIAL_LEARNING_RATE,
    min_lr: float = MIN_LEARNING_RATE,
) -> float:
    cosine = 0.5 * (1.0 + math.cos(math.pi * epoch_in_cycle / max(cycle_length, 1e-12)))
    return min_lr + (initial_lr - min_lr) * cosine


class CosineAnnealingWarmRestarts(tf.keras.callbacks.Callback):

    def __init__(
        self,
        first_restart_epoch: int,
        cycle_multiplier: float,
        initial_learning_rate: float = INITIAL_LEARNING_RATE,
        min_learning_rate: float = MIN_LEARNING_RATE,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.min_learning_rate = min_learning_rate
        self.cycle_multiplier = cycle_multiplier
        self.verbose = verbose
        self.epoch_in_cycle = 0
        self.cycle_length = float(first_restart_epoch)
        self.lr_history: list[float] = []

    def on_epoch_begin(self, epoch, logs=None) -> None:
        lr = cosine_annealing_lr(
            self.epoch_in_cycle,
            self.cycle_length,
            self.initial_learning_rate,
            self.min_learning_rate,
        )
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self.lr_history.append(lr)
        if self.verbose:
            print(f"epoch {epoch + 1}: lr={lr:.6f}")

    def on_epoch_end(self, epoch, logs=None) -> None:
        self.epoch_in_cycle += 1
        if self.epoch_in_cycle >= self.cycle_length:
            self.epoch_in_cycle = 0
            self.cycle_length *= self.cycle_multiplier


def schedule_preview(
    epochs: int,
    first_restart_epoch: int,
    cycle_multiplier: float,
    initial_learning_rate: float = INITIAL_LEARNING_RATE,
    min_learning_rate: float = MIN_LEARNING_RATE,
) -> list[float]:
    values: list[float] = []
    epoch_in_cycle = 0
    cycle_length = float(first_restart_epoch)
    for _ in range(epochs):
        values.append(
            cosine_annealing_lr(
                epoch_in_cycle, cycle_length, initial_learning_rate, min_learning_rate
            )
        )
        epoch_in_cycle += 1
        if epoch_in_cycle >= cycle_length:
            epoch_in_cycle = 0
            cycle_length *= cycle_multiplier
    return values
