"""

    Base Trainer


"""
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import (  # noqa
    _METRIC_COLLECTION as METRIC_COLLECTION,
)


class BaseTrainer(pl.LightningModule):
    def __init__(self, base_dir: str, start_time: str) -> None:
        super().__init__()
        self.base_dir = base_dir
        self.start_time = start_time

    @property
    def name(self) -> str:
        return "output"

    @property
    def storage(self) -> Path:
        directory = Path(self.base_dir).joinpath(self.name).joinpath(self.start_time)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def checkpoints_dir(self) -> Path:
        directory = self.storage.joinpath("checkpoints")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def samples_dir(self) -> Path:
        directory = self.storage.joinpath("samples")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def image_samples_dir(self) -> Path:
        directory = self.samples_dir.joinpath("images")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def audio_samples_dir(self) -> Path:
        directory = self.samples_dir.joinpath("audio")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def analyses_dir(self) -> Path:
        directory = self.samples_dir.joinpath("analyses")
        directory.mkdir(parents=True, exist_ok=True)
        return directory
