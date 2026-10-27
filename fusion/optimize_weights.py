from __future__ import annotations

import argparse
import json

import numpy as np

from common.labels import FOREGROUND_CLASSES
from fusion.classwise import (
    BACKGROUND_MODE,
    MAX_ITERATIONS,
    optimize_classwise_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-2d", required=True, help=".npy validation probabilities, 2D branch")
    parser.add_argument("--pred-3d", required=True, help=".npy validation probabilities, 3D branch")
    parser.add_argument("--target", required=True, help=".npy one-hot validation ground truth")
    parser.add_argument("--output", required=True, help="destination JSON for the weights")
    parser.add_argument("--classes", type=int, nargs="*", default=list(FOREGROUND_CLASSES))
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument(
        "--background-mode", choices=("fuse", "zero", "from_2d"), default=BACKGROUND_MODE
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p2d = np.load(args.pred_2d)
    p3d = np.load(args.pred_3d)
    target = np.load(args.target)

    weights, summary = optimize_classwise_weights(
        p2d,
        p3d,
        target,
        foreground_classes=tuple(args.classes),
        max_iterations=args.max_iterations,
        background_mode=args.background_mode,
    )
    weights.save(args.output)

    print(json.dumps({"weights": weights.as_mapping(), **summary}, indent=2))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
