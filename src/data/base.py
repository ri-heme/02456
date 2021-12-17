__all__ = ["DataModule"]

from typing import Callable, Dict

from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule


class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset_factory: Callable,
        train_size: float = 0.8,
        val_size: float = 0.15,
        batch_size=32,
        num_processes=0,
    ):
        super().__init__()
        if train_size + val_size > 1.0:
            raise ValueError("Size of train and test split should not exceed 1.0.")
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.full_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
        self.dataset_factory = dataset_factory

    @property
    def num_features(self) -> int:
        return self.full_dataset.num_features

    @property
    def num_classes(self) -> int:
        return self.full_dataset.num_classes

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return self.full_dataset.idx_to_class

    def setup(self, stage: str = None) -> None:
        self.full_dataset = self.dataset_factory()
        full_size = len(self.full_dataset)
        train_size = int(full_size * self.train_size)
        val_size = int(full_size * self.val_size)
        test_size = full_size - train_size - val_size
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(
            self.full_dataset, (train_size, test_size, val_size)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=self.num_processes,
            persistent_workers=self.num_processes > 0,
            shuffle=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            self.batch_size,
            num_workers=self.num_processes,
            persistent_workers=self.num_processes > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=self.num_processes,
            persistent_workers=self.num_processes > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.full_dataset, 300)
