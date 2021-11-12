__all__ = ["SnpDataset"]

import pickle
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

from src._typing import PathLike, Tuple

_METADATA_COLUMNS = [
    "index",
    "index2",
    "encoding",
    "encoding2",
    "time_num",
    "time_str",
    "country",
    "SNP_cov",
    "SNP_cov2",
    "SNP_cov3",
    "ancestry",
]


class SnpDataset(Dataset):
    """SNP dataset.

    Attributes
    ----------
    n_features : int
        Number of input features
    n_classes : int
        Number of output classes
    """

    def __init__(self, raw_data_path) -> None:
        self._read_data(raw_data_path)

    def _read_data(self, raw_data_path: PathLike) -> None:
        raw_data_path = Path(raw_data_path)

        try:
            encodings_path = next(raw_data_path.joinpath("tensor_data").glob("*.json"))
        except StopIteration:
            raise FileNotFoundError("Encodings file not found.")

        with open(encodings_path, "rb") as file:
            encodings = pickle.load(file)

        try:
            metadata_path = next(raw_data_path.joinpath("metadata").glob("*.tsv"))
        except StopIteration:
            raise FileNotFoundError("Encodings file not found.")

        metadata = (
            pd.read_csv(metadata_path, sep="\t", header=None, names=_METADATA_COLUMNS)
            .fillna("Unknown")
            .where(lambda df: df.encoding.isin(encodings))
            .dropna()
        )

        input_files = sorted(raw_data_path.joinpath("tensor_data").glob("*.pt"))
        if len(input_files) != metadata.shape[0]:
            raise Exception("Number of tensor files does not match number of targets.")

        self.encoder = OneHotEncoder(sparse=False)
        targets = self.encoder.fit_transform(metadata.ancestry.values.reshape(-1, 1))

        self.samples = list(zip(input_files, targets))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        X, y = self.samples[idx]
        X = torch.Tensor(torch.load(X)[:, [0]])  # Second dimension is not needed
        X = torch.nan_to_num(X, nan=0).view(1, -1)  # Convert NaNs to zeros
        y = torch.LongTensor(y).view(-1)
        return X, y

    @property
    def n_features(self) -> int:
        return self[0][0].shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.encoder.categories_[0])

    @property
    def idx_to_class(self):
        return {
            tuple([0.0] * i + [1.0] + [0.0] * (self.n_target_classes - i - 1)): cat
            for i, cat in enumerate(self.encoder.categories_[0])
        }
