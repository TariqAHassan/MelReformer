"""

    Transformer Model

"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange

from melreformer.core.sequence.transformer.components.pos_enc import PositionalEncoding
from melreformer.core.sequence.transformer.decoder import Decoder
from melreformer.core.sequence.transformer.encoder import Encoder


class EncoderHead(nn.Module):
    def __init__(self, encoder_block_size: int, mode: str = "linear") -> None:
        super().__init__()
        self.encoder_block_size = encoder_block_size
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] > self.encoder_block_size:
            x = F.interpolate(x, size=self.encoder_block_size, mode=self.mode)
        return x.transpose(1, 2)


class Transformer(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        window_size: int = 32,
        num_blocks: int = 6,
        num_heads: int = 8,
        dim_embed: int = 512 + 256,
        encoder_block_size: int = 256,
        decoder_block_size: int = 64,
        dim_ff: Optional[int] = None,
        p_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.window_size = window_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dim_embed = dim_embed
        self.encoder_block_size = encoder_block_size
        self.decoder_block_size = decoder_block_size
        self.dim_ff = dim_ff or (dim_embed * 2)  # See StyleGAN3 paper for motivation
        self.p_dropout = p_dropout

        self.start_token = nn.Parameter(torch.randn(1, 1, self.dim_input))

        self.encoder = nn.Sequential(
            EncoderHead(encoder_block_size),
            nn.Linear(n_mels, dim_embed),
            PositionalEncoding(
                dim_embed=dim_embed,
                p_dropout=p_dropout,
                max_len=encoder_block_size,
            ),
            Encoder(
                num_blocks=num_blocks,
                num_heads=num_heads,
                dim_embed=dim_embed,
                dim_ff=self.dim_ff,
                p_dropout=p_dropout,
            ),
        )

        # Decoder
        self.preprocess_decoder = nn.Sequential(
            nn.Linear(self.dim_input, dim_embed),
            PositionalEncoding(
                dim_embed=dim_embed,
                p_dropout=p_dropout,
                max_len=decoder_block_size,
            ),
        )
        self.decoder = Decoder(
            num_blocks=num_blocks,
            num_heads=num_heads,
            dim_embed=dim_embed,
            block_size=decoder_block_size,
            dim_ff=self.dim_ff,
            p_dropout=p_dropout,
        )
        self.head = nn.Linear(dim_embed, self.dim_input)

    @property
    def dim_input(self) -> int:
        return self.n_mels * self.window_size

    def _add_start_token(self, y: torch.Tensor) -> torch.Tensor:
        s = self.start_token.repeat(y.shape[0], 1, 1)
        return torch.cat([s, y], dim=1)

    def _fold(self, x: torch.Tensor) -> torch.Tensor:
        folded = [c.flatten(1) for c in x.split(self.window_size, dim=-1)]
        return torch.stack(folded, dim=1)

    def _unfold(self, y: torch.Tensor) -> torch.Tensor:
        unfolded = [c.view(-1, self.n_mels, self.window_size) for c in y.unbind(1)]
        return torch.cat(unfolded, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        fold_input: bool = True,
        unfold_output: bool = True,
    ) -> torch.Tensor:
        y = self._fold(x) if fold_input else x
        y = self.decoder(self.preprocess_decoder(y), cond=cond)
        y = self.head(y)
        return self._unfold(y) if unfold_output else y

    def train_forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._fold(x)
        y = self._add_start_token(y)[:, :-1, :]
        return self.forward(y, cond=self.encoder(x), fold_input=False)

    @torch.inference_mode()
    def generate(
        self,
        cond: torch.Tensor,
        prime: Optional[torch.Tensor] = None,
        n_steps: int = 1,
        fold_prime: bool = True,
        verbose: bool = False,
    ) -> torch.Tensor:
        cond = self.encoder(cond)
        if prime is None:
            y = self.start_token.repeat(cond.shape[0], 1, 1)
        else:
            y = self._fold(prime) if fold_prime else prime

        for _ in trange(n_steps, disable=not verbose, desc="Generating"):
            # 1. Crop sequence to `decoder_block_size` if it's too long.
            input_fwd = (
                y
                if y.shape[1] <= self.decoder_block_size
                else y[:, -self.decoder_block_size :]
            )
            # 2. Compute the next embedding
            next_sample = self.forward(
                input_fwd,
                cond=cond,
                fold_input=False,
                unfold_output=False,
            )[:, -1, :]
            # 3. Append the new embedding (autoregression)
            y = torch.cat([y, next_sample.unsqueeze(1)], dim=1)
        return self._unfold(y[:, 1:, :] if prime is None else y)


if __name__ == "__main__":
    from melreformer.utils.general import Timer
    from melreformer.utils.training import count_parameters

    PRIME_STEPS = 8

    self = tformer = Transformer()
    print(f"Transformer Params: {count_parameters(tformer):,}")

    # Test Prediction
    x = cond = torch.randn(1, tformer.n_mels, 2048)

    assert (self._unfold(self._fold(x)) == x).all().item()

    with Timer("Transformer.encoder()", print_on_enter=False):
        e = self.encoder(x)
    assert e.shape == (x.shape[0], tformer.encoder_block_size, tformer.dim_embed)

    with Timer("Transformer.forward()", print_on_enter=False):
        out = tformer.forward(x, cond=e)
    assert out.shape == x.shape

    with Timer("Transformer.train_forward()", print_on_enter=False):
        out = tformer.train_forward(x)
    assert out.shape == x.shape

    # Test Generation
    with Timer("Transformer.generate()", print_on_enter=False):
        g = tformer.generate(
            cond=cond,
            prime=x[..., : (PRIME_STEPS * self.window_size)] if PRIME_STEPS else None,
            n_steps=tformer.decoder_block_size - PRIME_STEPS,
            fold_prime=True,
            verbose=True,
        )
    assert g.shape == x.shape
