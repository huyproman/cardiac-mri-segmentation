from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from common.geometry import BBox, clip_bbox

DETECTOR_IMGSZ = 352
DETECTION_CONF_THRESHOLD = 0.25
DETECTION_IOU_THRESHOLD = 0.5


@dataclass(frozen=True)
class Detection:

    bbox: BBox
    confidence: float
    class_id: int = 0

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)


def slice_to_detector_image(slice_2d: np.ndarray) -> np.ndarray:
    if slice_2d.ndim != 2:
        raise ValueError(f"expected a 2D slice, got shape {slice_2d.shape}")
    finite = np.nan_to_num(slice_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(finite.min()), float(finite.max())
    scaled = np.zeros_like(finite) if hi <= lo else (finite - lo) / (hi - lo)
    gray = (scaled * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def parse_ultralytics_result(result: Any, shape: tuple[int, int]) -> list[Detection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = np.asarray(boxes.xyxy, dtype=np.float32).reshape(-1, 4)
    conf = np.asarray(getattr(boxes, "conf", np.ones(len(xyxy))), dtype=np.float32).reshape(-1)
    cls = np.asarray(getattr(boxes, "cls", np.zeros(len(xyxy))), dtype=np.int32).reshape(-1)

    detections: list[Detection] = []
    for i in range(xyxy.shape[0]):
        col0, row0, col1, row1 = xyxy[i]
        bbox = clip_bbox((int(row0), int(col0), int(row1), int(col1)), shape)
        detections.append(Detection(bbox=bbox, confidence=float(conf[i]), class_id=int(cls[i])))
    return detections


def select_detection(
    detections: Sequence[Detection],
    min_confidence: float = DETECTION_CONF_THRESHOLD,
) -> Detection | None:
    eligible = [d for d in detections if d.confidence >= min_confidence]
    if not eligible:
        return None
    return max(eligible, key=lambda d: (d.confidence, d.area))


class YoloDetector:

    def __init__(
        self,
        weights: str,
        imgsz: int = DETECTOR_IMGSZ,
        confidence_threshold: float = DETECTION_CONF_THRESHOLD,
        iou_threshold: float = DETECTION_IOU_THRESHOLD,
    ) -> None:
        from ultralytics import YOLO

        self.imgsz = imgsz
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(weights)

    def detect_slice(self, slice_2d: np.ndarray) -> list[Detection]:
        image = slice_to_detector_image(slice_2d)
        results = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        if not results:
            return []
        return parse_ultralytics_result(results[0], slice_2d.shape)

    def detect_volume(self, volume: np.ndarray) -> list[list[Detection]]:
        return [self.detect_slice(volume[:, :, z]) for z in range(volume.shape[2])]
