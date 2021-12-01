__all__ = []

from pathlib import Path

import pytest

import src
from src import data

EXPECTED_SAMPLES = 6155
EXPECTED_CLASSES = 13
EXPECTED_FEATURES = 328044
EXPECTED_TRAIN_SIZE = int(EXPECTED_SAMPLES * 0.7)
EXPECTED_TEST_SIZE = int(EXPECTED_SAMPLES * 0.2)
EXPECTED_VAL_SIZE = EXPECTED_SAMPLES - EXPECTED_TRAIN_SIZE - EXPECTED_TEST_SIZE
EXPECTED_BATCH_SIZE = 64


def test_dataset():
    raw_data_path = Path(__file__).parent.parent / "data" / "raw"
    dataset = src.data.SNPDataset(raw_data_path)
    assert len(dataset) == EXPECTED_SAMPLES
    assert dataset.num_features == EXPECTED_FEATURES
    assert dataset.num_classes == EXPECTED_CLASSES


@pytest.fixture(scope="class")
def snp_datamodule(request):
    request.cls.datamodule = src.data.SNPDataModule(
        train_size=0.7, test_size=0.2, batch_size=64
    )
    request.cls.datamodule.setup()
    yield


@pytest.mark.usefixtures("snp_datamodule")
class TestDataModule:
    @pytest.mark.parametrize(
        "dataloader",
        ["train_dataloader", "test_dataloader", "val_dataloader", "predict_dataloader"],
    )
    def test_batch_size(self, dataloader):
        batch = next(iter(getattr(self.datamodule, dataloader)()))
        assert len(batch) == 2
        assert all(x.size(0) == EXPECTED_BATCH_SIZE for x in batch)

    def test_split(self):
        assert len(self.datamodule.train_data) == EXPECTED_TRAIN_SIZE
        assert len(self.datamodule.test_data) == EXPECTED_TEST_SIZE
        assert len(self.datamodule.val_data) == EXPECTED_VAL_SIZE
