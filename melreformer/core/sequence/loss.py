"""

    Standard Loss

"""
from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.nn import functional as F


class ReconstructionLoss:  # LeastSquaresLoss
    def __init__(self, log_fn: Optional[Callable[..., Any]] = None) -> None:
        self.log_fn = log_fn

    def log(self, name: str, value: str, prog_bar: bool = False) -> None:
        if callable(self.log_fn):
            self.log_fn(name, value, prog_bar=prog_bar)  # noqa

    @staticmethod
    def _compute_fake_loss(fake_preds: torch.Tensor) -> torch.Tensor:
        return torch.mean(fake_preds**2)

    @staticmethod
    def _compute_real_loss(real_preds: torch.Tensor) -> torch.Tensor:
        return torch.mean((1 - real_preds) ** 2)

    def generator_loss(
        self,
        real_batch: torch.Tensor,
        fake_batch: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Reconstruction Loss
        l1_loss = F.l1_loss(fake_batch, target=real_batch)
        self.log("g_l1_loss", value=l1_loss, prog_bar=True)
        return l1_loss
