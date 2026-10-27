from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.nifti import iter_acdc_processed
from common.patient_splits import patient_folds
from roi_localization.detection import (
    DETECTION_IOU_THRESHOLD,
    DETECTOR_IMGSZ,
)
from roi_localization.yolo_dataset import export_records, write_data_yaml

BASE_MODEL = "yolo12m.pt"
EPOCHS = 150
PATIENCE = 50
BATCH = 16
DEVICE = "0"
RUN_NAME = "roi_detect"


def train_detector(
    data_yaml: str | Path,
    base_model: str = BASE_MODEL,
    epochs: int = EPOCHS,
    patience: int = PATIENCE,
    batch: int = BATCH,
    device: str = DEVICE,
    imgsz: int = DETECTOR_IMGSZ,
    project: str | None = None,
    run_name: str = RUN_NAME,
) -> Any:
    from ultralytics import YOLO

    model = YOLO(base_model)
    return model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        batch=batch,
        device=device,
        imgsz=imgsz,
        project=project,
        name=run_name,
    )


def validate_detector(
    weights: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    imgsz: int = DETECTOR_IMGSZ,
    iou_threshold: float = DETECTION_IOU_THRESHOLD,
) -> Any:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        conf=0.001,
        iou=iou_threshold,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--train-split", default="training")
    parser.add_argument("--test-split", default="testing")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--imgsz", type=int, default=DETECTOR_IMGSZ)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--project")
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)

    records = list(iter_acdc_processed(args.input_root, args.train_split, require_mask=True))
    if not records:
        raise SystemExit(f"no annotated volumes found in {args.input_root}/{args.train_split}")

    folds = patient_folds([r.patient_id for r in records])
    val_patients = set(folds[0].val_patients)
    train_records = [r for r in records if r.patient_id not in val_patients]
    val_records = [r for r in records if r.patient_id in val_patients]

    print("train slices:", export_records(train_records, root / "train"))
    print("val slices:", export_records(val_records, root / "val"))

    test_dir = None
    test_records = list(iter_acdc_processed(args.input_root, args.test_split, require_mask=True))
    if test_records:
        print("test slices:", export_records(test_records, root / "test"))
        test_dir = "test"

    data_yaml = write_data_yaml(root, "train", "val", test_dir)
    print("wrote", data_yaml)

    if args.export_only:
        return
    train_detector(
        data_yaml,
        base_model=args.base_model,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        imgsz=args.imgsz,
        project=args.project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
