"""

    Encoder

"""
import torch
from torch import nn

from melreformer.core.sequence.transformer.components.attn import MultiHeadAttention
from melreformer.core.sequence.transformer.components.feedforward import (
    PositionwiseFeedForward,
)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_embed: int,
        dim_ff: int,
        p_dropout: float,
        causal: bool = False,  # OK if the encoder looks forward in time
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_ff = dim_ff
        self.p_dropout = p_dropout
        self.causal = causal

        self.sublayer0 = nn.Sequential(
            nn.LayerNorm(dim_embed),
            MultiHeadAttention(
                num_heads=num_heads,
                dim_embed=dim_embed,
                p_dropout=p_dropout,
                causal=causal,
            ),
        )
        self.sublayer1 = nn.Sequential(
            nn.LayerNorm(dim_embed),
            PositionwiseFeedForward(
                d_in=dim_embed,
                d_hidden=dim_ff,
                p_dropout=p_dropout,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.sublayer0(x) + x
        x = self.sublayer1(x) + x
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        dim_embed: int,
        dim_ff: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.dim_ff = dim_ff
        self.p_dropout = p_dropout

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    num_heads=num_heads,
                    dim_embed=dim_embed,
                    dim_ff=dim_ff,
                    p_dropout=p_dropout,
                    causal=False,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(dim_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.layer_norm(x)
