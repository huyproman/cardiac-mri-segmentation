from __future__ import annotations

import argparse
import json

import numpy as np

from common.labels import CLASS_NAMES, FOREGROUND_CLASSES
from evaluation.metrics import HD95_SPACING, evaluate_volume, mean_dice, mean_hd95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", required=True, help=".npy integer label volume")
    parser.add_argument("--target", required=True, help=".npy integer label volume")
    parser.add_argument("--classes", type=int, nargs="*", default=list(FOREGROUND_CLASSES))
    parser.add_argument("--spacing", type=float, nargs=2, default=HD95_SPACING)
    parser.add_argument("--output-json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction = np.load(args.prediction)
    target = np.load(args.target)
    metrics = evaluate_volume(
        prediction, target, classes=tuple(args.classes), spacing=tuple(args.spacing)
    )

    payload = {
        "per_class": {
            CLASS_NAMES[m.class_id]: {
                "dice": m.dice,
                "hd95_mean": m.hd95_mean,
                "hd95_std": m.hd95_std,
            }
            for m in metrics
        },
        "mean_dice": mean_dice(metrics),
        "mean_hd95": mean_hd95(metrics),
    }

    print(json.dumps(payload, indent=2))
    if args.output_json:
        with open(args.output_json, "w") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
