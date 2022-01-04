__all__ = ["LatentDataset", "LatentDataModule"]

from pathlib import Path
from functools import partial
from typing import Tuple, Dict

import pandas as pd
import torch
from dotenv import find_dotenv
from torch.utils.data import Dataset

from src.data.base import DataModule
from src.data.snp import SNPDataset
from src.visualization.projection import PROJECTION_CSV_FILENAME


def find_projection_filepath(name: str, version: str) -> Path:
    return (
        Path(find_dotenv()).parent / "models" / name / version / PROJECTION_CSV_FILENAME
    )


class LatentDataset(Dataset):
    def __init__(self, model_name: str, version: str) -> None:
        super().__init__()
        projection_filepath = find_projection_filepath(model_name, version)
        self.samples = pd.read_csv(projection_filepath, index_col=0)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        sample = self.samples.iloc[idx]
        return torch.Tensor(sample.values), torch.LongTensor([sample.name])

    @property
    def num_features(self) -> int:
        return self.samples.shape[1]

    @property
    def num_classes(self) -> int:
        return self.samples.index.max() + 1

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return SNPDataset().idx_to_class


class LatentDataModule(DataModule):
    """Latent data module. Provides consistent data loaders for training,
    validation, and inference. Each sample consists of a low-dimensional
    (latent) representation of the SNP variants in an individual's allele and
    an encoded label corresponding to the individual's ancestry.

    Attributes
    ----------
    num_features : int
        Number of input features
    num_classes : int
        Number of target classes
    idx_to_class : dict
        Mapping of index to target class

    Parameters
    ----------
    model_name : str
        Name of the model used to create the latent representation
    version : str
        Experiment version of the model
    """
    def __init__(
        self,
        model_name: str,
        version: str,
        train_size: float = 0.8,
        val_size: float = 0.15,
        batch_size=32,
        num_processes=0,
    ):
        super().__init__(
            partial(LatentDataset, model_name=model_name, version=version),
            train_size,
            val_size,
            batch_size,
            num_processes,
        )
