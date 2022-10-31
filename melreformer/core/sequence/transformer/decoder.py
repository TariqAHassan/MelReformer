"""

    Decoder

"""
import torch
from torch import nn

from melreformer.core.sequence.transformer.components.attn import MultiHeadAttention
from melreformer.core.sequence.transformer.components.feedforward import (
    PositionwiseFeedForward,
)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_embed: int,
        block_size: int,
        dim_ff: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.block_size = block_size
        self.dim_ff = dim_ff
        self.p_dropout = p_dropout

        # Self-attention
        self.ln_sa = nn.LayerNorm(dim_embed)
        self.self_attn = MultiHeadAttention(
            num_heads=num_heads,
            dim_embed=dim_embed,
            block_size=block_size,
            p_dropout=p_dropout,
            causal=True,
        )

        # 2. Cross Attention
        self.ln_ca = nn.LayerNorm(dim_embed)
        self.cross_attn = MultiHeadAttention(
            num_heads=num_heads,
            dim_embed=dim_embed,
            p_dropout=p_dropout,
            causal=False,  # the model can access any time here
        )

        # 3. Feed Forward
        self.ln_ff = nn.LayerNorm(dim_embed)
        self.ff = PositionwiseFeedForward(
            d_in=dim_embed,
            d_hidden=dim_ff,
            p_dropout=p_dropout,
        )

    def sublayer1(self, x: torch.Tensor) -> torch.Tensor:
        return self.self_attn(self.ln_sa(x))

    def sublayer2(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(self.ln_ca(x), x_kv=cond)

    def sublayer3(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(self.ln_ff(x))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.sublayer1(x) + x
        x = self.sublayer2(x, cond=cond) + x
        x = self.sublayer3(x) + x
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        dim_embed: int,
        block_size: int,
        dim_ff: int,
        p_dropout: float,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.block_size = block_size
        self.dim_ff = dim_ff
        self.p_dropout = p_dropout

        self.blocks: list[DecoderBlock] = nn.ModuleList(
            [
                DecoderBlock(
                    num_heads=num_heads,
                    dim_embed=dim_embed,
                    block_size=block_size,
                    dim_ff=dim_ff,
                    p_dropout=p_dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.layer_norm = nn.LayerNorm(dim_embed)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, cond)
        return self.layer_norm(x)


if __name__ == "__main__":
    self = Decoder(
        dim_embed=512,
        num_blocks=6,
        num_heads=8,
        block_size=64,
        dim_ff=512,
        p_dropout=0.1,
    )

    x = torch.randn(1, 32, self.dim_embed)
    out = self(x, x)
