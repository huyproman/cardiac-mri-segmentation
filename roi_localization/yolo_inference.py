from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.geometry import ROI_CROP_SIZE
from common.nifti import iter_acdc_processed, save_nifti
from roi_localization.detection import (
    DETECTION_CONF_THRESHOLD,
    DETECTION_IOU_THRESHOLD,
    DETECTOR_IMGSZ,
    YoloDetector,
)
from roi_localization.detector_roi import extract_roi_from_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=DETECTOR_IMGSZ)
    parser.add_argument("--conf", type=float, default=DETECTION_CONF_THRESHOLD)
    parser.add_argument("--iou", type=float, default=DETECTION_IOU_THRESHOLD)
    parser.add_argument("--crop-size", type=int, nargs=2, default=ROI_CROP_SIZE)
    parser.add_argument(
        "--prune-slices",
        action="store_true",
        help="drop slices with weak detector evidence outside the heart-bearing range",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = YoloDetector(args.weights, args.imgsz, args.conf, args.iou)
    out_root = Path(args.output_root) / args.split

    for record in iter_acdc_processed(args.input_root, args.split, require_mask=False):
        detections = detector.detect_volume(record.image)
        roi = extract_roi_from_detections(
            record.image,
            detections,
            record.patient_id,
            record.phase,
            crop_size=tuple(args.crop_size),
            slice_indices=record.slice_indices,
            prune_slices=args.prune_slices,
            min_confidence=args.conf,
        )

        target = out_root / record.patient_id
        stem = f"{record.patient_id}_{record.phase}"
        save_nifti(target / f"{stem}_roi.nii.gz", roi.image, record.affine)
        (target / f"{stem}_crops.json").write_text(
            json.dumps(
                {
                    "patient_id": roi.patient_id,
                    "phase": roi.phase,
                    "slice_indices": list(roi.slice_indices),
                    "original_depth": int(record.num_slices),
                    "crops": [c.to_dict() for c in roi.crops],
                },
                indent=2,
            )
        )
        print(f"{record.patient_id} {record.phase}: {roi.num_slices}/{record.num_slices} slices")


if __name__ == "__main__":
    main()
