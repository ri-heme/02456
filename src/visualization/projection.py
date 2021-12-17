__all__ = ["plot_projection"]

from os import PathLike
from pathlib import Path
from typing import Dict, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.manifold import TSNE

from src.models.logger import CSVLogger

PROJECTION_FILENAME = "projection.png"


def plot_projection(
    logger_or_path: Union[CSVLogger, PathLike],
    y: Union[np.ndarray, torch.Tensor],
    z: np.ndarray,
    idx_to_class: Dict[int, str],
    use_tsne: bool = False,
):
    """Generates a projection plot.

    Parameters
    ----------
    logger_or_path : src.models.logger.CSVLogger or os.PathLike
        Logger object or file path to save plot as
    y : np.ndarray or torch.Tensor
        Targets vector
    z : np.ndarray
        Low-dimensional representation of features
    idx_to_class : dict
        Mapping of target indices to labels
    use_tsne : bool, optional
        Whether to use t-SNE to further reduce the number of dimensions of z to
        2, by default False
    """
    if use_tsne:
        z = TSNE(n_components=2, init="pca")

    if isinstance(logger_or_path, CSVLogger):
        projection_filepath = Path(logger_or_path.log_dir, PROJECTION_FILENAME)
    else:
        projection_filepath = logger_or_path

    fig, (ax, legend_ax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]})
    palette = sns.color_palette("Paired", 13)

    if isinstance(y, torch.Tensor):
        y = y.numpy()
    y = y.reshape(-1)

    for idx in range(y.max()):
        plot_data = z[y == idx, :]
        ax.scatter(
            plot_data[:, 0],
            plot_data[:, 1],
            color=palette[idx],
            label=idx_to_class[idx],
        )

    handles, labels = ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="center", borderaxespad=0)
    legend_ax.axis("off")

    ax.set_xlabel("dim0")
    ax.set_ylabel("dim1")
    ax.set_title("z")

    fig.tight_layout()
    fig.savefig(projection_filepath, bbox_inches="tight")
