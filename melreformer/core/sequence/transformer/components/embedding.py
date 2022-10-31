"""

    Embedding

"""
from math import sqrt

import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * sqrt(self.dim)
