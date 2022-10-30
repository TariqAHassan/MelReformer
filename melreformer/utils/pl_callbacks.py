"""

    Callbacks

"""
from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar


class CustomTqdmProgressBar(TQDMProgressBar):
    def __init__(
        self,
        ignored_metrics: tuple[str, ...] | None = ("v_num",),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ignored_metrics = set(ignored_metrics or [])

    def get_metrics(
        self,
        trainer: pl.Trainer,
        model: pl.LightningModule,
    ) -> dict[str, int | str]:
        items = super().get_metrics(trainer, model)
        for m in self.ignored_metrics:
            items.pop(m, None)
        return items


class CustomModelCheckpoint(ModelCheckpoint):
    def _epoch_save_override(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        pl_module.save_samples()
        if pl_module.score_this_epoch():
            pl_module.score_generator()
            monitor_candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, monitor_candidates)
            self._save_last_checkpoint(trainer, monitor_candidates)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        self._epoch_save_override(trainer, pl_module)

    def on_validation_end(
        self,
        trainer: "pl.Trainer",
        pl_module: pl.LightningModule,
    ) -> None:
        self._epoch_save_override(trainer, pl_module)
