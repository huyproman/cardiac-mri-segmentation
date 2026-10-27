from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from roi_localization.detection import DETECTION_CONF_THRESHOLD, Detection

MIN_BOX_AREA_FRACTION = 0.01
MAX_INTERIOR_GAP = 2


@dataclass(frozen=True)
class SliceEvidence:
    index: int
    confidence: float
    area_fraction: float
    weak: bool


def _best(detections: Sequence[Detection]) -> Detection | None:
    return max(detections, key=lambda d: d.confidence) if detections else None


def score_slices(
    detections_per_slice: Sequence[Sequence[Detection]],
    image_area: int,
    min_confidence: float = DETECTION_CONF_THRESHOLD,
    min_box_area_fraction: float = MIN_BOX_AREA_FRACTION,
) -> list[SliceEvidence]:
    evidence: list[SliceEvidence] = []
    for z, dets in enumerate(detections_per_slice):
        best = _best(dets)
        confidence = best.confidence if best else 0.0
        area_fraction = (best.area / image_area) if (best and image_area > 0) else 0.0
        weak = area_fraction < min_box_area_fraction and confidence < min_confidence
        evidence.append(SliceEvidence(z, confidence, area_fraction, weak))
    return evidence


def contiguous_heart_range(
    evidence: Sequence[SliceEvidence],
    max_interior_gap: int = MAX_INTERIOR_GAP,
) -> tuple[int, int] | None:
    strong = [e.index for e in evidence if not e.weak]
    if not strong:
        return None

    best: tuple[int, int] | None = None
    start = prev = strong[0]
    for index in strong[1:]:
        if index - prev - 1 <= max_interior_gap:
            prev = index
            continue
        if best is None or (prev - start) > (best[1] - best[0]):
            best = (start, prev)
        start = prev = index
    if best is None or (prev - start) > (best[1] - best[0]):
        best = (start, prev)
    return best


def select_inference_slices(
    detections_per_slice: Sequence[Sequence[Detection]],
    image_shape: tuple[int, int],
    min_confidence: float = DETECTION_CONF_THRESHOLD,
    min_box_area_fraction: float = MIN_BOX_AREA_FRACTION,
    max_interior_gap: int = MAX_INTERIOR_GAP,
) -> tuple[int, ...]:
    n = len(detections_per_slice)
    if n == 0:
        return ()

    image_area = int(image_shape[0]) * int(image_shape[1])
    evidence = score_slices(
        detections_per_slice, image_area, min_confidence, min_box_area_fraction
    )
    heart_range = contiguous_heart_range(evidence, max_interior_gap)
    if heart_range is None:
        return tuple(range(n))

    lo, hi = heart_range
    return tuple(e.index for e in evidence if (lo <= e.index <= hi) or not e.weak)
