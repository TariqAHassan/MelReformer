"""

    Losses

"""
from typing import Any, Callable, Optional

from torch import nn


class BaseAdversarialLoss:
    def __init__(
        self,
        discriminator: nn.Module,
        log_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.discriminator = discriminator
        self.log_fn = log_fn

    def log(self, value: str, name: str, prog_bar: bool = False) -> None:
        if callable(self.log_fn):
            self.log_fn(name, value, prog_bar=prog_bar)  # noqa
