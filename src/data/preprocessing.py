__all__ = ["SnpDataset"]

import pickle
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
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
        Number of target classes
    idx_to_class : dict
        Mapping of index to target class
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

        input_files = sorted(
            raw_data_path.joinpath("tensor_data").glob("*.pt"),
            key=lambda path: int(re.search("\d+", path.stem)[0]),
        )
        if len(input_files) != metadata.shape[0]:
            raise Exception("Number of tensor files does not match number of targets.")

        self.encoder = LabelEncoder()
        targets = self.encoder.fit_transform(metadata.ancestry.values)

        self.samples = list(zip(input_files, targets))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        X, y = self.samples[idx]
        X = torch.Tensor(torch.load(X).T)  # Second dimension is not needed
        X = torch.nan_to_num(X, nan=0)  # Convert NaNs to zeros
        y = torch.LongTensor([y])
        return X, y

    @property
    def n_features(self) -> int:
        return self[0][0].numel()

    @property
    def n_classes(self) -> int:
        return len(self.encoder.classes_)

    @property
    def idx_to_class(self) -> dict:
        return dict(zip(range(self.n_classes), self.encoder.classes_))
