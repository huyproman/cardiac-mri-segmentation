from __future__ import annotations

import argparse
from pathlib import Path

from common.labels import NUM_CLASSES
from common.nifti import iter_acdc_processed
from common.patient_splits import iter_folds
from roi_localization.groundtruth_roi import extract_roi_from_groundtruth
from segmentation.branch_3d.model import attention_unet_3d
from segmentation.callbacks import (
    SEED,
    build_callbacks,
    checkpoint_exists,
    enable_mixed_precision,
    fold_checkpoint_path,
    set_global_seed,
)
from segmentation.datasets import make_volume_dataset, stack_volumes
from segmentation.losses import dice_metric, make_hybrid_loss
from segmentation.schedules import INITIAL_LEARNING_RATE

TARGET_DEPTH = 10
INPUT_SHAPE = (224, 224, TARGET_DEPTH, 1)
CHANNELS = 32
BATCH_SIZE = 2
EPOCHS = 210
FIRST_RESTART_EPOCH = 30
CYCLE_MULTIPLIER = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--channels", type=int, default=CHANNELS)
    parser.add_argument("--target-depth", type=int, default=TARGET_DEPTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--folds", type=int, nargs="*", help="subset of fold indices to run")
    return parser.parse_args()


def main() -> None:
    import tensorflow as tf

    args = parse_args()
    set_global_seed(args.seed)
    if not args.no_mixed_precision:
        enable_mixed_precision(True)

    records = list(iter_acdc_processed(args.input_root, args.split, require_mask=True))
    if not records:
        raise SystemExit(f"no annotated volumes in {args.input_root}/{args.split}")
    roi_volumes = [extract_roi_from_groundtruth(r) for r in records]

    strategy = tf.distribute.MirroredStrategy()
    print(f"replicas: {strategy.num_replicas_in_sync}")

    input_shape = (INPUT_SHAPE[0], INPUT_SHAPE[1], args.target_depth, 1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold, train_volumes, val_volumes in iter_folds(roi_volumes, seed=args.seed):
        if args.folds and fold.index not in args.folds:
            continue
        print(
            f"fold {fold.index}: {len(fold.train_patients)} train / "
            f"{len(fold.val_patients)} val patients"
        )

        x_train, y_train, _, _ = stack_volumes(train_volumes, args.target_depth)
        x_val, y_val, _, _ = stack_volumes(val_volumes, args.target_depth)

        train_ds = make_volume_dataset(
            x_train, y_train, args.batch_size, augment=True, seed=args.seed
        )
        val_ds = make_volume_dataset(
            x_val, y_val, args.batch_size, augment=False, seed=args.seed, shuffle=False
        )

        with strategy.scope():
            model = attention_unet_3d(input_shape, NUM_CLASSES, args.channels)
            model.compile(
                optimizer=tf.keras.optimizers.Adam(INITIAL_LEARNING_RATE),
                loss=make_hybrid_loss(),
                metrics=[dice_metric()],
            )
        checkpoint_path = fold_checkpoint_path(output_dir, fold.index)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            callbacks=build_callbacks(
                checkpoint_path, FIRST_RESTART_EPOCH, CYCLE_MULTIPLIER
            ),
        )
        if not checkpoint_exists(checkpoint_path):
            raise SystemExit(f"fold {fold.index}: no checkpoint at {checkpoint_path}")
        model.load_weights(str(checkpoint_path))


if __name__ == "__main__":
    main()
