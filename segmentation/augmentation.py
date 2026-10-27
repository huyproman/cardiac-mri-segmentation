from __future__ import annotations

import numpy as np

FLIP_PROB = 0.5
TRANSLATION_PROB = 0.5
TRANSLATION_RANGE = (-8.0, 8.0)
DEPTH_SHIFT_PROB = 0.5
DEPTH_SHIFT_RANGE = (-4, 4)
GAMMA_PROB = 0.5
GAMMA_RANGE = (0.85, 1.15)
BRIGHTNESS_PROB = 0.5
BRIGHTNESS_RANGE = (-0.05, 0.05)
CONTRAST_PROB = 0.5
CONTRAST_RANGE = (0.95, 1.05)


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    return float(rng.uniform(bounds[0], bounds[1]))


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    lo, hi = float(image.min()), float(image.max())
    if hi <= lo:
        return image
    scaled = (image - lo) / (hi - lo)
    return (np.clip(scaled, 1e-7, 1.0) ** gamma) * (hi - lo) + lo


def _apply_brightness(image: np.ndarray, delta: float) -> np.ndarray:
    return image + delta


def _apply_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    mean = float(image.mean())
    return mean + factor * (image - mean)


def _shift_2d(array: np.ndarray, dx: int, dy: int, fill: float = 0.0) -> np.ndarray:
    out = np.full_like(array, fill)
    h, w = array.shape[:2]
    src_x0, dst_x0 = (0, dx) if dx >= 0 else (-dx, 0)
    src_y0, dst_y0 = (0, dy) if dy >= 0 else (-dy, 0)
    nx = h - abs(dx)
    ny = w - abs(dy)
    if nx <= 0 or ny <= 0:
        return out
    out[dst_x0 : dst_x0 + nx, dst_y0 : dst_y0 + ny] = array[
        src_x0 : src_x0 + nx, src_y0 : src_y0 + ny
    ]
    return out


def augment_slice(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator | None = None,
    flip_prob: float = FLIP_PROB,
    translation_prob: float = TRANSLATION_PROB,
    translation_range: tuple[float, float] = TRANSLATION_RANGE,
    gamma_prob: float = GAMMA_PROB,
    gamma_range: tuple[float, float] = GAMMA_RANGE,
    brightness_prob: float = BRIGHTNESS_PROB,
    brightness_range: tuple[float, float] = BRIGHTNESS_RANGE,
    contrast_prob: float = CONTRAST_PROB,
    contrast_range: tuple[float, float] = CONTRAST_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    image = np.array(image, dtype=np.float32, copy=True)
    mask = np.array(mask, copy=True)

    if rng.random() < flip_prob:
        image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
    if rng.random() < flip_prob:
        image, mask = np.flip(image, axis=0).copy(), np.flip(mask, axis=0).copy()
    if rng.random() < translation_prob:
        dx = int(round(_uniform(rng, translation_range)))
        dy = int(round(_uniform(rng, translation_range)))
        image = _shift_2d(image, dx, dy, fill=0.0)
        mask = _shift_2d(mask, dx, dy, fill=0)

    if rng.random() < gamma_prob:
        image = _apply_gamma(image, _uniform(rng, gamma_range))
    if rng.random() < brightness_prob:
        image = _apply_brightness(image, _uniform(rng, brightness_range))
    if rng.random() < contrast_prob:
        image = _apply_contrast(image, _uniform(rng, contrast_range))

    return image.astype(np.float32), mask


def augment_volume(
    image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator | None = None,
    flip_prob: float = FLIP_PROB,
    depth_shift_prob: float = DEPTH_SHIFT_PROB,
    depth_shift_range: tuple[int, int] = DEPTH_SHIFT_RANGE,
    gamma_prob: float = GAMMA_PROB,
    gamma_range: tuple[float, float] = GAMMA_RANGE,
    brightness_prob: float = BRIGHTNESS_PROB,
    brightness_range: tuple[float, float] = BRIGHTNESS_RANGE,
    contrast_prob: float = CONTRAST_PROB,
    contrast_range: tuple[float, float] = CONTRAST_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    image = np.array(image, dtype=np.float32, copy=True)
    mask = np.array(mask, copy=True)

    if rng.random() < flip_prob:
        image, mask = np.flip(image, axis=1).copy(), np.flip(mask, axis=1).copy()
    if rng.random() < flip_prob:
        image, mask = np.flip(image, axis=0).copy(), np.flip(mask, axis=0).copy()
    if rng.random() < depth_shift_prob:
        lo, hi = depth_shift_range
        shift = int(rng.integers(int(lo), int(hi) + 1))
        if shift:
            image = np.roll(image, shift, axis=2)
            mask = np.roll(mask, shift, axis=2)

    if rng.random() < gamma_prob:
        image = _apply_gamma(image, _uniform(rng, gamma_range))
    if rng.random() < brightness_prob:
        image = _apply_brightness(image, _uniform(rng, brightness_range))
    if rng.random() < contrast_prob:
        image = _apply_contrast(image, _uniform(rng, contrast_range))

    return image.astype(np.float32), mask
