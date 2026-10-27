from __future__ import annotations

from typing import Iterable

import numpy as np

from common.labels import NUM_CLASSES
from preprocessing.resampling import PAD_SLICE, crop_or_pad_depth, depth_transform_indices
from segmentation.augmentation import augment_slice, augment_volume
from segmentation.callbacks import SEED


def one_hot(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return np.eye(num_classes, dtype=np.float32)[np.asarray(labels, dtype=np.int64)]


def stack_slices(roi_volumes: Iterable) -> tuple[np.ndarray, np.ndarray, list[str]]:
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    patients: list[str] = []
    for volume in roi_volumes:
        if volume.mask is None:
            raise ValueError(f"{volume.patient_id} {volume.phase} has no mask")
        for z in range(volume.image.shape[2]):
            images.append(volume.image[:, :, z][..., None])
            labels.append(volume.mask[:, :, z])
            patients.append(volume.patient_id)
    if not images:
        raise ValueError("no slices to stack")
    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.uint8),
        patients,
    )


def stack_volumes(
    roi_volumes: Iterable, target_depth: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, ...]]]:
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    patients: list[str] = []
    depth_maps: list[tuple[int, ...]] = []
    for volume in roi_volumes:
        if volume.mask is None:
            raise ValueError(f"{volume.patient_id} {volume.phase} has no mask")
        images.append(crop_or_pad_depth(volume.image, target_depth)[..., None])
        labels.append(crop_or_pad_depth(volume.mask, target_depth))
        patients.append(volume.patient_id)
        local = depth_transform_indices(volume.image.shape[2], target_depth)
        source = volume.slice_indices or tuple(range(volume.image.shape[2]))
        depth_maps.append(
            tuple(PAD_SLICE if i == PAD_SLICE else int(source[i]) for i in local)
        )
    if not images:
        raise ValueError("no volumes to stack")
    return (
        np.asarray(images, dtype=np.float32),
        np.asarray(labels, dtype=np.uint8),
        patients,
        depth_maps,
    )


def _build_dataset(samples, labels, batch_size, num_classes, augment, augment_fn, seed, shuffle):
    import tensorflow as tf

    mask_shape = labels.shape[1:] + (num_classes,)

    def to_sample(sample: np.ndarray, label: np.ndarray, epoch, index):
        if augment:
            rng = np.random.default_rng([int(seed), int(epoch), int(index)])
            augmented, label = augment_fn(sample[..., 0], label, rng)
            sample = augmented[..., None]
        return sample.astype(np.float32), one_hot(label, num_classes)

    dataset = tf.data.Dataset.from_tensor_slices((samples, labels)).enumerate()

    epoch_counter = tf.Variable(0, dtype=tf.int64, trainable=False)

    def tag_epoch(index, element):
        epoch = tf.cond(
            tf.equal(index, 0),
            lambda: epoch_counter.assign_add(1).value(),
            lambda: epoch_counter.value(),
        )
        return index, epoch, element

    dataset = dataset.map(tag_epoch, num_parallel_calls=None, deterministic=True)

    def wrapper(index, epoch, element):
        sample, label = element
        out = tf.numpy_function(
            to_sample, [sample, label, epoch, index], [tf.float32, tf.float32]
        )
        out[0].set_shape(samples.shape[1:])
        out[1].set_shape(mask_shape)
        return out[0], out[1]

    dataset = dataset.map(wrapper, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if shuffle:
        dataset = dataset.shuffle(len(samples), seed=seed)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_slice_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_classes: int = NUM_CLASSES,
    augment: bool = False,
    seed: int = SEED,
    shuffle: bool = True,
):
    return _build_dataset(
        images, labels, batch_size, num_classes, augment, augment_slice, seed, shuffle
    )


def make_volume_dataset(
    volumes: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    num_classes: int = NUM_CLASSES,
    augment: bool = False,
    seed: int = SEED,
    shuffle: bool = True,
):
    return _build_dataset(
        volumes, labels, batch_size, num_classes, augment, augment_volume, seed, shuffle
    )
