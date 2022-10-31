"""

    Attn

    Notes:
        * `causal=True` assumes that the number of timesteps in x and y are the same.

    References:
        * https://kikaben.com/transformers-coding-details/

"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _make_mask(t0: int, t1: int) -> Tensor:
    return torch.tril(torch.ones(t0, t1)).view(1, 1, t0, t1)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        dim_embed: int,
        block_size: int | None = None,
        p_dropout: float = 0.0,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.block_size = block_size
        self.p_dropout = p_dropout
        self.causal = causal

        self.dim_head, remainder = divmod(self.dim_embed, self.num_heads)
        if remainder:
            raise ValueError("Embed <> Heads Mismatch")
        elif causal and block_size is None:
            raise ValueError("Block size must be specified for causal masking")

        self.q = nn.Linear(dim_embed, dim_embed)
        self.k = nn.Linear(dim_embed, dim_embed)
        self.v = nn.Linear(dim_embed, dim_embed)

        if causal:
            self.register_buffer("mask", _make_mask(block_size, block_size))

        self.output = nn.Sequential(
            nn.Linear(dim_embed, dim_embed),
            nn.Dropout(p_dropout),
        )

    def _compute_attn(self, q: Tensor, k: Tensor, v: Tensor, time: int) -> Tensor:
        scores = (q @ k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        if self.causal:
            scores = scores.masked_fill(
                self.mask[:, :, :time, :time] == 0, float("-inf")
            )
        return scores.softmax(dim=-1) @ v

    def forward(self, x_q: Tensor, x_kv: Tensor | None = None) -> Tensor:
        # `x_q` is the input from which queries will be derived from, whereas
        # `x_kv` will be used to derive the keys and values. If the latter is
        # not specified, `x_q` will be used to derive queries, keys and values.
        B, T, _ = x_q.shape
        if x_kv is None:
            x_kv = x_q

        q = self.q(x_q).view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        k = self.k(x_kv).view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        v = self.v(x_kv).view(B, -1, self.num_heads, self.dim_head).transpose(1, 2)
        return self.output(
            self._compute_attn(q, k=k, v=v, time=T)
            .transpose(1, 2)
            .contiguous()
            .view(B, -1, self.dim_embed)
        )


if __name__ == "__main__":
    mha = MultiHeadAttention(
        dim_embed=512,
        num_heads=8,
        block_size=64,
        p_dropout=0.1,
        causal=True,
    )

    # Test Causal
    x = torch.randn(1, mha.block_size, mha.dim_embed)
    out0 = mha(x)
    assert out0.shape == x.shape

    mha.causal = False
    y = torch.randn(1, mha.block_size // 4, mha.dim_embed)
    out1 = mha(x, y)
    assert out1.shape == x.shape
