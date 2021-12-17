__all__ = ["generate_projection", "plot_projection"]

from os import PathLike
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl
from sklearn.manifold import TSNE

from src.models.extraction.base import BaseVAE
from src.models.logger import CSVLogger

PROJECTION_CSV_FILENAME = "projection.csv"
PROJECTION_IMG_FILENAME = "projection.png"
PROJECTION_IMG_ALL_FILENAME = "projection_all.png"


def generate_projection(
    logger_or_path: Union[CSVLogger, PathLike],
    model: BaseVAE,
    datamodule: pl.LightningDataModule,
    sample_size: int = 300,
    use_tsne: bool = False,
) -> None:
    """Generates a projection for every sample (as a CSV file) in the dataset
    and plots all and a random subset of projections.

    Parameters
    ----------
    logger_or_path : src.models.logger.CSVLogger or os.PathLike
        Logger object or file path to save projections CSV as
    model : BaseVAE
        Trained feature extraction model
    datamodule : pytorch_lightning.LightningDataModule
        Data module, where to obtain a predict dataloader
    sample_size : int, optional
        Number of samples for the sample plot, by default 300
    use_tsne : bool, optional
        Whether to use t-SNE to further reduce the number of dimensions of the
        projections to 2, by default False
    """
    if isinstance(logger_or_path, CSVLogger):
        csv_filepath = Path(logger_or_path.log_dir, PROJECTION_CSV_FILENAME)
    else:
        csv_filepath = Path(logger_or_path)

    dataloader = datamodule.predict_dataloader()
    data = []
    colnames = [f"z{i}" for i in range(model.hparams.latent_features)]
    for x, y in dataloader:
        z = model.project(x)
        df = pd.DataFrame(z, columns=colnames, index=y.numpy().flatten())
        data.append(df)
    data = pd.concat(data)
    data.index.name = "y"
    data.to_csv(csv_filepath)

    img_filepath = csv_filepath.parent / PROJECTION_IMG_ALL_FILENAME
    plot_projection(
        img_filepath, data.index.values, data.values, datamodule.idx_to_class, use_tsne
    )

    img_filepath = csv_filepath.parent / PROJECTION_IMG_FILENAME
    sample = data.sample(sample_size)
    plot_projection(
        img_filepath,
        sample.index.values,
        sample.values,
        datamodule.idx_to_class,
        use_tsne,
    )


def plot_projection(
    logger_or_path: Union[CSVLogger, PathLike],
    y: Union[np.ndarray, torch.Tensor],
    z: np.ndarray,
    idx_to_class: Dict[int, str],
    use_tsne: bool = False,
) -> None:
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
        z = TSNE(n_components=2, init="pca").fit_transform(z)

    if isinstance(logger_or_path, CSVLogger):
        projection_filepath = Path(logger_or_path.log_dir, PROJECTION_IMG_FILENAME)
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
