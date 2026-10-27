from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from common.labels import NUM_CLASSES

STAGE_TITLES = ("MRI", "Ground Truth", "2D branch", "3D branch", "Fused", "Refined")


def _to_labels(array: np.ndarray) -> np.ndarray:
    return np.argmax(array, axis=-1) if array.ndim == 3 else array


def plot_pipeline_stages(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction_2d: np.ndarray,
    prediction_3d: np.ndarray,
    fused: np.ndarray,
    refined: np.ndarray,
    num_classes: int = NUM_CLASSES,
    save_path: str | Path | None = None,
    titles: Sequence[str] = STAGE_TITLES,
):
    import matplotlib

    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        image,
        _to_labels(ground_truth),
        _to_labels(prediction_2d),
        _to_labels(prediction_3d),
        _to_labels(fused),
        _to_labels(refined),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, panel, title in zip(axes, panels, titles):
        if title == titles[0]:
            ax.imshow(panel, cmap="gray")
        else:
            ax.imshow(panel, cmap="jet", vmin=0, vmax=num_classes - 1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    return fig
