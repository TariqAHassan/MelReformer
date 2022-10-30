"""

    Embedding

"""
import torch
from torch import nn
from math import sqrt


class Embedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * sqrt(self.dim)
