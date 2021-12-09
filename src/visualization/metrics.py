__all__ = ["plot_metrics"]

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.logger import CSVLogger
from src._typing import Tuple


METRICS_FIG_FILENAME = "metrics.png"


def plot_metrics(logger: CSVLogger, figsize: Tuple[int]):
    """Plots training/validation metrics.

    Parameters
    ----------
    logger : src.models.CSVLogger
        Logger object used to train model
    figsize : tuple of int
        Tuple of plot's (width, height) in inches
    """
    metrics = pd.read_csv(
        logger.experiment.metrics_file_path, index_col=[0, 1], header=[0, 1]
    )
    hparams = logger.experiment.hparams
    colnames = tuple(metrics.columns.get_level_values(0).unique())

    fig, axs = plt.subplots(ncols=len(colnames), figsize=figsize)
    for ax, colname in zip(axs, colnames):
        plot_data = metrics[colname].reset_index(level=1, drop=True)
        ax.set_ylabel(colname)
        sns.lineplot(data=plot_data, ax=ax)

    hparams_fmt = str(hparams).replace("'", "")[1:-1]
    title = f"{logger.name}, {hparams_fmt}"
    fig.suptitle(title)
    fig.savefig(Path(logger.log_dir, METRICS_FIG_FILENAME))
