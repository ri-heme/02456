__all__ = ["plot_metrics", "plot_grid"]

from collections import Iterable
from math import sqrt, ceil
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_lightning.core.saving import load_hparams_from_yaml
from pytorch_lightning.loggers.csv_logs import ExperimentWriter

from src.models.logger import CSVLogger


METRICS_FIG_FILENAME = "metrics.png"


def _load_hparams(experiment_path: PathLike) -> dict:
    return load_hparams_from_yaml(
        Path(experiment_path, ExperimentWriter.NAME_HPARAMS_FILE)
    )


def _load_metrics(experiment_path: PathLike) -> pd.DataFrame:
    return pd.read_csv(
        Path(experiment_path, ExperimentWriter.NAME_METRICS_FILE),
        index_col=[0, 1],
        header=[0, 1],
    )


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
    metrics = _load_metrics(experiment_path)
    hparams = _load_hparams(experiment_path)
    colnames = tuple(metrics.columns.get_level_values(0).unique())

    fig, axs = plt.subplots(ncols=len(colnames), figsize=figsize)
    if not isinstance(axs, Iterable):
        axs = [axs]
    for ax, colname in zip(axs, colnames):
        plot_data = metrics[colname].reset_index(level=1, drop=True)
        ax.set_ylabel(colname)
        sns.lineplot(data=plot_data, ax=ax)

    hparams_fmt = str(hparams).replace("'", "")[1:-1]
    title = f"{experiment_path.parent.name}, {hparams_fmt}"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(Path(experiment_path, METRICS_FIG_FILENAME), bbox_inches="tight")


def _format_hparams(experiment_path: Path):
    hparams = _load_hparams(experiment_path)
    hparams_fmt = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
    return f"{experiment_path.parent.name} ({experiment_path.name})\n{hparams_fmt}"


def plot_grid(
    model_path_or_experiment_paths: Union[Path, List[Path]],
    metric: str = "loss",
    figsize: Tuple[int] = (10, 10),
    title_height: float = 0.2,
):
    """Plots a grid comparing a metric between experiments.

    Parameters
    ----------
    model_path_or_experiment_paths : pathlib.Path or list of pathlib.Path
        Path containing multiple experiments or list of paths pointing to
        individual experiments
    metric : str, optional
        Name of the metric to compare, by default "loss"
    figsize : tuple of int
        Tuple of plot's dimensions (width, height) in inches, by default
        (10, 10)
    title_height : float, optional
        Height of a plot's title relative to plot, by default 0.2
    """
    if isinstance(model_path_or_experiment_paths, Path):
        # If model directory is given, find all experiments (i.e., versions)
        experiment_paths = [
            path
            for path in model_path_or_experiment_paths.glob("*")
            if path.is_dir()
        ]
    elif isinstance(model_path_or_experiment_paths, list):
        # Specific versions were given
        experiment_paths = model_path_or_experiment_paths
    else:
        raise ValueError("Unexpected input given.")

    # Calculate number of subplots
    n = len(experiment_paths)
    ncols = ceil(sqrt(n))
    nrows = ceil(n / ncols)
    height_ratios = (title_height, 1.0) * nrows

    fig = plt.figure(figsize=figsize)
    # Duplicate nrows to have one row for title and one for plot
    gs = fig.add_gridspec(nrows * 2, ncols, height_ratios=height_ratios)

    shared_ax = None

    for i in range(gs._nrows):
        for j in range(gs._ncols):
            k = (i // 2) * nrows + j
            ax = fig.add_subplot(gs[i, j], sharey=shared_ax)
            if i % 2 == 0:
                # Even rows are titles
                hparams = _format_hparams(experiment_paths[k])
                ax.axis("off")
                ax.text(
                    0.5,
                    0.05,
                    hparams,
                    ha="center",
                    va="center",
                    ma="left",
                    transform=ax.transAxes,
                )
            else:
                # Odd rows are plots
                metrics = _load_metrics(experiment_paths[k])
                plot_data = metrics[metric].reset_index(level=1, drop=True)
                sns.lineplot(data=plot_data, ax=ax)
                # Add/hide y label depending on column
                if j == 0:
                    ax.set_ylabel(metric)
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                # Have following plots share axis ticks 
                if shared_ax is None:
                    shared_ax = ax

    gs.tight_layout(fig)
    fig.savefig(Path(experiment_paths[k].parent, f"{metric}.png"), bbox_inches="tight")

