"""

    Feed Forward Layer

"""
import torch
from torch import nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, p_dropout: float = 0.1) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.p_dropout = p_dropout

        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout),
            nn.Linear(d_hidden, d_in),
            nn.Dropout(p_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
