__all__ = ["CSVLogger"]

from pathlib import Path
from datetime import datetime

import pandas as pd
from pytorch_lightning.loggers import CSVLogger as LightningCSVLogger
from dotenv import find_dotenv

from src._typing import List


class CSVLogger(LightningCSVLogger):
    """CSV logger that uses dates for versioning.
    
    Parameters
    ----------
    name : str
        Experiment name.
    metrics : list of str
        List of metrics tracked by this logger. Used to create the CSV header.
    """
    def __init__(
        self, name: str, metrics: List[str] = ["loss"]
    ):
        save_dir = Path(find_dotenv()).parent / "models"
        super().__init__(save_dir, name=name, version=None)
        self.columns = pd.MultiIndex.from_product([["val", "train"], metrics])

    def finalize(self, status: str) -> None:
        self.save()
        # Output CSV files have one row for each mode: training/validation
        # Training rows have NaN values in validation columns, and vice versa
        # So this is meant to condense the table by combining rows
        # 1. Read CSV
        metrics = pd.read_csv(self.experiment.metrics_file_path)
        # 2. Split into train/val dataframes
        train_metrics = metrics.dropna(subset=["train_loss"]).set_index(
            ["epoch", "step"]
        )
        val_metrics = metrics.dropna(subset=["val_loss"]).set_index(["epoch", "step"])
        # 3. Combine
        metrics = train_metrics.combine_first(val_metrics)
        # 4. Set header => metrics * (train, val)
        metrics.columns = self.columns
        metrics = metrics.swaplevel(axis=1)
        # 5. Overwrite
        metrics.to_csv(self.experiment.metrics_file_path)

    def _get_next_version(self) -> str:
        date = datetime.now()
        return f"version_{date.year}-{date.month}-{date.day}_{date.hour}-{date.minute}"
