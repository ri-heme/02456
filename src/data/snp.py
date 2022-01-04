__all__ = ["SNPDataset", "SNPDataModule"]

import pickle
import re
from os import PathLike
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from dotenv import find_dotenv
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

from src.data.base import DataModule

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


def find_raw_data_path() -> PathLike:
    return Path(find_dotenv()).parent / "data" / "raw"


class SNPDataset(Dataset):
    """SNP dataset.

    Attributes
    ----------
    num_features : int
        Number of input features
    num_classes : int
        Number of target classes
    sample_shape : (int, int)
        Shape of a sample
    idx_to_class : dict
        Mapping of index to target class
    """

    def __init__(self) -> None:
        self._read_data(find_raw_data_path())

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
            .set_index("encoding")
            .loc[encodings]
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

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, ...]:
        X, y = self.samples[idx]
        # Encode:
        # [0, 0] -> [1, 0, 0, 0] 0 SNPs
        # [0, 1] -> [0, 1, 0, 0] NaNs
        # [1, 0] -> [0, 0, 1, 0] 1 SNP
        # [1, 1] -> [0, 0, 0, 1] 2 SNPs (not used)
        X = torch.Tensor([2, 1]) * torch.Tensor(torch.load(X))
        X = X.sum(axis=1).nan_to_num(nan=1).long()
        X = one_hot(X).float().T
        y = torch.LongTensor([y])
        return X, y

    @property
    def sample_shape(self) -> Tuple[int]:
        return tuple(self[0][0].shape)

    @property
    def num_features(self) -> int:
        return self[0][0].numel()

    @property
    def num_classes(self) -> int:
        return len(self.encoder.classes_)

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return dict(zip(range(self.num_classes), self.encoder.classes_))


class SNPDataModule(DataModule):
    """SNP data module. Provides consistent data loaders for training,
    validation, and inference. Each sample consists of an encoded label
    corresponding to an ancestry category and a set of features representing SNP
    variants. Each SNP variant has three elements representing whether it was
    present or not in one allele of the individual or its data is missing due to
    a sequencing error. Therefore, a sample with 3 SNP variants would have 9
    features (3 elements per SNP).

    Attributes
    ----------
    num_features : int
        Number of input features
    num_classes : int
        Number of target classes
    sample_shape : (int, int)
        Shape of a sample
    idx_to_class : dict
        Mapping of index to target class
    """
    def __init__(
        self,
        train_size: float = 0.8,
        val_size: float = 0.15,
        batch_size=32,
        num_processes=0,
    ):
        super().__init__(SNPDataset, train_size, val_size, batch_size, num_processes)

    @property
    def sample_shape(self) -> Tuple[int]:
        return self.full_dataset.sample_shape
