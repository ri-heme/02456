__all__ = ["plot_metrics"]

from os import PathLike
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.loggers.csv_logs import ExperimentWriter

from src.models.logger import CSVLogger


METRICS_FIG_FILENAME = "metrics.png"


def plot_metrics(
    logger_or_path: Union[CSVLogger, PathLike], figsize: Tuple[int]
) -> None:
    """Plots training/validation metrics.

    Parameters
    ----------
    logger_or_path : src.models.CSVLogger or os.PathLike
        Logger object used to train model or path to experiment results
    figsize : tuple of int
        Tuple of plot's (width, height) in inches
    """
    experiment_path = (
        logger_or_path
        if not isinstance(logger_or_path, CSVLogger)
        else Path(logger_or_path.log_dir)
    )
    metrics = pd.read_csv(
        Path(experiment_path, ExperimentWriter.NAME_METRICS_FILE),
        index_col=[0, 1],
        header=[0, 1],
    )
    hparams = load_hparams_from_yaml(
        Path(experiment_path, ExperimentWriter.NAME_HPARAMS_FILE)
    )
    colnames = tuple(metrics.columns.get_level_values(0).unique())

    fig, axs = plt.subplots(ncols=len(colnames), figsize=figsize)
    for ax, colname in zip(axs, colnames):
        plot_data = metrics[colname].reset_index(level=1, drop=True)
        ax.set_ylabel(colname)
        sns.lineplot(data=plot_data, ax=ax)

    hparams_fmt = str(hparams).replace("'", "")[1:-1]
    title = f"{experiment_path.parent.name}, {hparams_fmt}"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(Path(experiment_path, METRICS_FIG_FILENAME), bbox_inches="tight")
