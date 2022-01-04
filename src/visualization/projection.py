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

# Bokeh Libraries
from bokeh.plotting import figure, save
from bokeh.io import output_file, export_png
from bokeh.models import ColumnDataSource, CDSView, IndexFilter
from bokeh.models import HoverTool

PROJECTION_CSV_FILENAME = "projection.csv"
PROJECTION_IMG_FILENAME = "projection.png"
PROJECTION_IMG_ALL_FILENAME = "projection_all.png"


def generate_projection(
    logger_or_path: Union[CSVLogger, PathLike],
    model: BaseVAE,
    datamodule: pl.LightningDataModule,
    sample_size: int = 300,
    use_tsne: bool = False,
    interactive: bool = False,
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
    interactive : bool, optional
        Whether to plot the projections in an interactive way, by default False
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
        df = pd.DataFrame(
            z, columns=colnames, index=y.numpy().flatten())
        data.append(df)
    data = pd.concat(data)
    data.index.name = "y"
    data.to_csv(csv_filepath)

    if interactive:
        img_filepath = csv_filepath.parent / PROJECTION_IMG_ALL_FILENAME
        plot_interactive_projection(
            img_filepath,
            data.index.values,
            data.values,
            datamodule.idx_to_class,
            use_tsne
        )

        img_filepath = csv_filepath.parent / PROJECTION_IMG_FILENAME
        sample = data.sample(sample_size)
        plot_interactive_projection(
            img_filepath,
            sample.index.values,
            sample.values,
            datamodule.idx_to_class,
            use_tsne,
        )
    else:
        img_filepath = csv_filepath.parent / PROJECTION_IMG_ALL_FILENAME
        plot_projection(
            img_filepath,
            data.index.values,
            data.values,
            datamodule.idx_to_class,
            use_tsne
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
        projection_filepath = Path(
            logger_or_path.log_dir, PROJECTION_IMG_FILENAME
            )
    else:
        projection_filepath = logger_or_path

    fig, (ax, legend_ax) = plt.subplots(
        ncols=2, gridspec_kw={"width_ratios": [4, 1]})
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


def plot_interactive_projection(
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
        projection_filepath = Path(
            logger_or_path.log_dir,
            PROJECTION_IMG_FILENAME,
            )
    else:
        projection_filepath = logger_or_path

    if isinstance(y, torch.Tensor):
        y = y.numpy()
    y = y.reshape(-1)

    _interactive_projection(y, z, idx_to_class, projection_filepath)


def _interactive_projection(
    y: np.ndarray,
    z: np.ndarray,
    idx_to_class: Dict[int, str],
    projection_filepath: str,
):
    """Bokeh interactive projection

    Parameters
    ----------
    y : np.ndarray
        Targets vactor
    z : np.ndarray
        Low-dimensional representation of features
    idx_to_class : dict
        Mapping of target indices to labels
    projection_filepath : string
        Filepath
    """
    output_file(f'{projection_filepath}.html')

    projection = pd.DataFrame(z, columns=["dim0", "dim1"])
    projection["y"] = y

    color = sns.color_palette('pastel', 13).as_hex()
    colors = []
    ancestry = []
    for row in projection.iterrows():
        colors.append(color[int(row[1]['y'])])
        ancestry.append(idx_to_class[int(row[1]['y'])])

    projection['colors'] = colors
    projection['ancestry'] = ancestry
    # Store the data in a ColumnDataSource
    projection_CDS = ColumnDataSource(projection)

    # Specify the selection tools to be made available
    select_tools = [
        'box_select',
        'lasso_select',
        'poly_select',
        'tap',
        'reset',
        'save',
        ]

    # Create the figure
    fig = figure(plot_height=400,
                 plot_width=600,
                 x_axis_label='dim0',
                 y_axis_label='dim1',
                 title='Projection',
                 toolbar_location='below',
                 tools=select_tools,)

    for sep_ancestry in sorted(list(set(ancestry))):
        indices = list(
            projection.index[projection['ancestry'] == sep_ancestry])
        view = CDSView(source=projection_CDS, filters=[IndexFilter(indices)])
        # Add square representing each metabolite
        fig.square(x='dim0',
                   y='dim1',
                   source=projection_CDS,
                   color='colors',
                   size=3,
                   selection_color='deepskyblue',
                   nonselection_color='lightgray',
                   nonselection_alpha=0.5,
                   legend_label=sep_ancestry,
                   view=view)

    # Format the tooltip
    tooltips = [
                ('Ancestry', '@ancestry'),
                ('Dim 0', '@dim0'),
                ('Dim 1', '@dim1'),
            ]

    # Configure a renderer to be used upon hover
    hover_glyph = fig.circle(x='dim0', y='dim1', source=projection_CDS,
                             size=15, alpha=0, color='lightpink',
                             selection_color='grey',
                             hover_fill_color='black', hover_alpha=0.5)

    # Add the HoverTool to the figure
    fig.add_tools(HoverTool(tooltips=tooltips, renderers=[hover_glyph]))

    fig.legend.click_policy = 'mute'
    fig.add_layout(fig.legend[0], 'right')

    # Visualize
    fig.background_fill_color = None
    fig.border_fill_color = None

    export_png(fig, filename=f'{projection_filepath}.png')
    save(fig)
