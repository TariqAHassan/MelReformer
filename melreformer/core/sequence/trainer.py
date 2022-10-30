"""

    Trainer

"""
from __future__ import annotations

import warnings
from argparse import Namespace
from typing import Any

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.types import (  # noqa
    _METRIC_COLLECTION as METRIC_COLLECTION,
)
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from torch.nn import functional as F

from melreformer.core.sequence.transformer.model import Transformer
from melreformer.core.sequence.loss import ReconstructionLoss

from melreformer.core.vocoder.trainer import HiFiTrainer
from melreformer.utils.audio import audio_saver, spec_grid_saver  # noqa
from melreformer.utils.general import get_git_metadata
from melreformer.utils.metrics import BestMetricTracker
from melreformer.utils.parsing import to_numeric
from melreformer.utils.training import compute_total_norm
from melreformer.core.base_trainer import BaseTrainer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=LightningDeprecationWarning)


class SequenceTrainer(BaseTrainer):
    def __init__(
        self,
        vocoder_path: str,
        n_segment: int,
        start_time: str,
        base_dir: str,
        lr: float = 3e-4,
        adam_betas: tuple[float, float] = (0.9, 0.99),
        prime_size: int = 9,
    ) -> None:
        super().__init__(
            base_dir=base_dir,
            start_time=start_time,
        )
        self.save_hyperparameters(
            Namespace(
                vocoder_path=vocoder_path,
                n_segment=n_segment,
                start_time=start_time,
                base_dir=base_dir,
                lr=lr,
                adam_betas=adam_betas,
                prime_size=prime_size,
            )
        )

        hifi: HiFiTrainer = HiFiTrainer.load_from_checkpoint(
            checkpoint_path=vocoder_path,
            map_location="cpu",
        )
        self.vocoder = hifi.g_ema
        self.melspec = hifi.melspec
        self.sample_rate = hifi.hparams.sample_rate

        self.g = Transformer(
            n_mels=hifi.melspec.n_mels,
            window_size=32,
            encoder_block_size=self.hparams.n_segment,
            decoder_block_size=64,
        )

        # Define loss ---
        self.loss = ReconstructionLoss(log_fn=self.log)

        # Tracked ---
        self.best_tracker = BestMetricTracker({"g_loss": "min", "g_loss_val": "min"})
        self.y_val = None

    @property
    def name(self) -> str:
        return "sequence_output"

    def log(self, name: str, value: METRIC_COLLECTION, **kwargs: Any) -> None:
        super().log(name, value=value, **kwargs)

        if name in self.best_tracker:
            value = to_numeric(value)
            if self.best_tracker.update(name, value, current_step=self.global_step):
                self.log(f"{name}_best", value=value)

    def on_train_start(self) -> None:
        if isinstance(self.logger, WandbLogger):
            git_metadata = get_git_metadata()
            if git_metadata:
                self.logger.experiment.config.update(git_metadata)  # noqa

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["misc"] = dict(
            git=get_git_metadata(),
            y_val=self.y_val.cpu(),
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.y_val = checkpoint["misc"]["y_val"]

    def _write_audio(self, samples: torch.Tensor, kind: str) -> None:
        audio_saver(
            tracks=self.vocoder(samples.squeeze(1)).cpu(),
            sample_rate=self.sample_rate,
            directory=self.audio_samples_dir.joinpath(kind).joinpath(
                f"epoch_{self.current_epoch}"
            ),
        )

    def _write_images(self, samples: torch.Tensor, kind: str) -> None:
        return spec_grid_saver(
            F.interpolate(
                (
                    torch.cat([self.y_val, samples], dim=1)
                    if kind == "recon"
                    else samples
                ).cpu(),
                size=1024,
                mode="linear",
            ),
            nrow=4,
            n_chunks=1,
            path=self.image_samples_dir.joinpath(
                f"epoch_{kind}_{self.current_epoch}.png"
            ),
            color_map="turbo",
        )

    def _write_samples(
        self,
        samples: torch.Tensor,
        kind: str,
        with_audio: bool = True,
    ) -> None:
        if with_audio:
            self._write_audio(samples, kind=kind)
        self._write_images(samples, kind=kind)

    @torch.inference_mode()
    def save_samples(self) -> None:
        # 1. Save Reconstructions
        self._write_samples(
            self.g.train_forward(self.y_val.to(self.device)),
            kind="recon",
            with_audio=False,
        )

        # 2. Save Conditioned Samples
        conditioned = self.g.generate(
            cond=self.y_val.to(self.device),
            n_steps=self.g.decoder_block_size,
        )
        self._write_samples(conditioned, kind="condn")

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        optimizer_idx: int = 0,
    ) -> torch.Tensor:
        mel = self.melspec(batch)
        fake = self.g.train_forward(mel)
        return self.loss.generator_loss(mel, fake_batch=fake)

    def validation_step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> torch.Tensor:
        mel = self.melspec(batch)
        if self.y_val is None:
            self.y_val = mel
        fake = self.g.train_forward(mel)
        return F.l1_loss(fake, target=mel)

    def validation_epoch_end(self, outputs: list[torch.Tensor]) -> None:
        self.log(f"g_loss_val", torch.stack(outputs).mean(), prog_bar=True)
        self.save_samples()

    def on_before_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int = 0,
    ) -> None:
        self.log(f"g_total_norm", compute_total_norm(self.g.parameters()))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt_g = torch.optim.AdamW(
            self.g.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.adam_betas,
        )
        return opt_g
