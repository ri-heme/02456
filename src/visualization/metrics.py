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

# Bokeh Libraries
from bokeh.layouts import gridplot
from bokeh.plotting import figure, save
from bokeh.io import output_file, export_png
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool

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
    logger_or_path: Union[CSVLogger, PathLike],
    figsize: Tuple[int],
    interactive: bool = False,
) -> None:
    """Plots training/validation metrics.

    Parameters
    ----------
    logger_or_path : src.models.CSVLogger or os.PathLike
        Logger object used to train model or path to experiment results
    figsize : tuple of int
        Tuple of plot's dimensions (width, height) in inches
    interactive : bool, optional
        Whether to show plot interactively, by default False
    """
    experiment_path = (
        Path(logger_or_path)
        if not isinstance(logger_or_path, CSVLogger)
        else Path(logger_or_path.log_dir)
    )
    if interactive:
        _interactive_metrics(experiment_path)
    else:
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
        fig.savefig(
            experiment_path / METRICS_FIG_FILENAME, bbox_inches="tight")


def _format_hparams(experiment_path: Path):
    hparams = _load_hparams(experiment_path)
    hparams_fmt = "\n".join([f"{k}: {v}" for k, v in hparams.items()])
    return f"{experiment_path.parent.name} \
        ({experiment_path.name})\n{hparams_fmt}"


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
            path for path in model_path_or_experiment_paths.glob("*")
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
    fig.savefig(
        Path(
            experiment_paths[k].parent, f"{metric}.png"), bbox_inches="tight"
            )


def _interactive_metrics(
    experiment_path: Path,
):
    """Interactively plot metrics for a given experiment.

    Parameters
    ----------
    experiment_path : pathlib.Path
        Path to experiment
    """

    data = pd.read_csv(
        Path(
            experiment_path,
            ExperimentWriter.NAME_METRICS_FILE
            ))[2:]
    data.reset_index(drop=True, inplace=True)
    if data.shape[1] == 4:
        acc = False
        data.rename(
            columns={
                'Unnamed: 0': 'epoch',
                'Unnamed: 1': 'step',
                'loss': 'val_loss',
                'loss.1': 'train_loss',
                }, inplace=True)
    else:
        acc = True
        data.rename(
            columns={
                'Unnamed: 0': 'epoch',
                'Unnamed: 1': 'step',
                'loss': 'val_loss',
                'acc': 'val_acc',
                'loss.1': 'train_loss',
                'acc.1': 'train_acc'
                }, inplace=True)
    data['epoch'] = data['epoch'].astype(int)

    output_file(experiment_path / 'metrics.html')

    # Store the data in a ColumnDataSource
    loss_acc = ColumnDataSource(data)

    # Specify the selection tools to be made available
    select_tools = [
        'box_select',
        'lasso_select',
        'poly_select',
        'tap',
        'reset',
        'save'
        ]

    # Create the figure
    fig = figure(plot_height=400,
                 plot_width=400,
                 x_axis_label='Epoch',
                 y_axis_label='Loss',
                 title='Loss',
                 toolbar_location='below',
                 tools=select_tools,
                 )

    # Add square representing each metabolite
    fig.line(
        'epoch',
        'val_loss',
        source=loss_acc,
        line_color="orange",
        legend_label="Validation Loss",
        line_width=2,
        )
    fig.line(
        'epoch',
        'train_loss',
        source=loss_acc,
        line_color="skyblue",
        legend_label="Training Loss",
        line_width=2,
        )

    # Format the tooltip
    tooltips = [
                ('Epoch', '@epoch'),
                ('val_loss', '@val_loss'),
                ('train_loss', '@train_loss'),
            ]

    # Configure a renderer to be used upon hover
    hover_glyph = fig.circle(
        x='epoch',
        y='train_loss',
        source=loss_acc,
        line_width=0,
        size=2,
        alpha=0,
        color='darkgrey',
        selection_color='grey',
        hover_fill_color='black',
        hover_alpha=0.5
        )
    hover_glyph_2 = fig.circle(
        x='epoch',
        y='val_loss',
        source=loss_acc,
        line_width=0,
        size=2,
        alpha=0,
        color='darkgrey',
        selection_color='grey',
        hover_fill_color='black',
        hover_alpha=0.5,
        )

    # Add the HoverTool to the figure
    fig.add_tools(
        HoverTool(
            tooltips=tooltips,
            renderers=[
                hover_glyph,
                hover_glyph_2
                ]
                ))

    if acc:
        # Create a figure relating the totals
        fig_2 = figure(
            plot_height=400,
            plot_width=400,
            x_axis_label='Epoch',
            y_axis_label='Accuracy',
            title='Accuracy',
            toolbar_location='below',
            tools=select_tools,
            )

        # Add square representing each metabolite
        fig_2.line(
            'epoch',
            'val_acc',
            source=loss_acc,
            line_color="orange",
            legend_label="Validation Accuracy",
            line_width=2,
            )
        fig_2.line(
            'epoch',
            'train_acc',
            source=loss_acc,
            line_color="skyblue",
            legend_label="Training Accuracy",
            line_width=2,
            )

        # Configure a renderer to be used upon hover
        acc_hover_glyph = fig_2.circle(
            x='epoch',
            y='train_acc',
            source=loss_acc,
            line_width=0,
            size=2,
            alpha=0,
            color='darkgrey',
            selection_color='grey',
            hover_fill_color='black',
            hover_alpha=0.5,
            )
        acc_hover_glyph_2 = fig_2.circle(
            x='epoch',
            y='val_acc',
            source=loss_acc,
            line_width=0,
            size=2,
            alpha=0,
            color='darkgrey',
            selection_color='grey',
            hover_fill_color='black',
            hover_alpha=0.5,
            )

        # Add the HoverTool to the figure
        fig_2.add_tools(
            HoverTool(
                tooltips=tooltips,
                renderers=[
                    acc_hover_glyph,
                    acc_hover_glyph_2]
                    ))

        fig_2.legend.location = "bottom_right"
        fig.legend.click_policy = 'mute'
        fig_2.legend.click_policy = 'mute'

        grid = gridplot([[fig, fig_2]])
    else:
        fig.legend.click_policy = 'mute'
        grid = gridplot([[fig]])

    # Visualize
    export_png(fig, filename=experiment_path / METRICS_FIG_FILENAME)
    save(grid)
