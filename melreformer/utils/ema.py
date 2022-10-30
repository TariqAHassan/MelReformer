"""

    EMA

"""
from __future__ import annotations

import torch


class ExponentialMovingAverage:
    def __init__(self, beta: float) -> None:
        self.beta = beta

        self.avg = None
        self.count = 0

    def __bool__(self) -> bool:
        return self.avg is not None

    def update(self, value: torch.Tensor) -> ExponentialMovingAverage:
        if self.avg is None:
            self.avg = (1 - self.beta) * value
        else:
            self.avg = self.avg * self.beta + (1 - self.beta) * value
        self.count += 1
        return self
