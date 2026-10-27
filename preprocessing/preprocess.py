from __future__ import annotations

import argparse
from pathlib import Path

from common.nifti import iter_acdc_processed, save_nifti
from preprocessing.resampling import (
    TARGET_IN_PLANE_SPACING,
    normalize_record,
    preprocess_volume,
)
from preprocessing.slice_pruning import (
    MAX_CONSECUTIVE_PARTIAL,
    MIN_CLASS_AREA,
    prune_volume,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target-spacing", type=float, nargs=2, default=TARGET_IN_PLANE_SPACING)
    parser.add_argument("--prune", action="store_true", help="apply mask-based slice pruning")
    parser.add_argument("--min-class-area", type=int, default=MIN_CLASS_AREA)
    parser.add_argument("--max-consecutive-partial", type=int, default=MAX_CONSECUTIVE_PARTIAL)
    parser.add_argument("--allow-missing-mask", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root) / args.split
    count = 0

    for record in iter_acdc_processed(
        args.input_root, args.split, require_mask=not args.allow_missing_mask
    ):
        processed = preprocess_volume(record, tuple(args.target_spacing), normalize=False)
        if args.prune:
            if processed.mask is None:
                raise SystemExit(f"--prune requires a mask; {processed.key} has none")
            processed = prune_volume(
                processed,
                min_class_area=args.min_class_area,
                max_consecutive_partial=args.max_consecutive_partial,
            )
        if processed.num_slices == 0:
            print(f"skip {processed.patient_id} {processed.phase}: no slices retained")
            continue
        processed = normalize_record(processed)

        target = out_root / processed.patient_id
        stem = f"{processed.patient_id}_{processed.phase}"
        save_nifti(target / f"{stem}_processed.nii.gz", processed.image, processed.affine)
        if processed.mask is not None:
            save_nifti(target / f"{stem}_gt_processed.nii.gz", processed.mask, processed.affine)
        count += 1
        print(f"{processed.patient_id} {processed.phase}: {processed.num_slices} slices")

    print(f"wrote {count} volumes to {out_root}")


if __name__ == "__main__":
    main()
