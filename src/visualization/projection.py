__all__ = ["plot_projection"]

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from src.models.logger import CSVLogger

PROJECTION_FILENAME = "projection.png"


def plot_projection(
    logger: CSVLogger, y: torch.Tensor, z: np.ndarray, idx_to_class: Dict[int, str]
):
    fig, ax = plt.subplots()
    palette = sns.color_palette("Set3", 13)

    y = y.numpy().flatten()
    for idx in range(y.max()):
        plot_data = z[y == idx, :]
        ax.scatter(
            plot_data[:, 0],
            plot_data[:, 1],
            color=palette[idx],
            label=idx_to_class[idx],
        )

    ax.set_xlabel("dim0")
    ax.set_ylabel("dim1")
    fig.suptitle("z")

    fig.savefig(Path(logger.log_dir, PROJECTION_FILENAME))
