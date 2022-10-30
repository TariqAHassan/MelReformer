"""

    Positional Encoder

"""
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        dim_embed: int,
        p_dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.dim_embed = dim_embed
        self.p_dropout = p_dropout
        self.max_len = max_len

        self.dropout = nn.Dropout(p=p_dropout)
        self.register_buffer("pe", self._build_pe())

    def _build_pe(self) -> torch.Tensor:
        position = torch.arange(self.max_len).unsqueeze(1)
        dim_pair = torch.arange(0, self.dim_embed, 2)
        div_term = torch.exp(dim_pair * (-math.log(10000.0) / self.dim_embed))

        pe = torch.zeros(self.max_len, self.dim_embed)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.shape[1]]
        return self.dropout(x)
