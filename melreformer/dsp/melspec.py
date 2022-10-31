"""

    Melspectrogram

"""
from functools import cached_property
from math import ceil, floor
from typing import Optional

import torch
from librosa.filters import mel as mel_filter
from torch import nn
from torch.nn import functional as F


def compress_drange(
    y: torch.Tensor,
    C: int = 1,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    return torch.log(torch.clamp(y, min=clip_val) * C)


class MelSpec(nn.Module):
    mel_basis: torch.Tensor
    stft_window: torch.Tensor

    def __init__(
        self,
        seq_len: int,
        n_mels: int = 128,
        n_fft: int = 1024,
        sr: int = 22_050,
        hop_length: int = 256,
        win_length: int = 1024,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        center: bool = False,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.sr = sr
        self.hop_length = hop_length
        self.win_length = win_length
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.eps = eps

        self.register_buffer(
            "stft_window",
            torch.hann_window(self.win_length),
        )
        self.register_buffer(
            "mel_basis",
            torch.from_numpy(
                mel_filter(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
            ),
        )

    @cached_property
    def shape(self) -> tuple[int, int]:
        y = self.forward(torch.randn(1, self.seq_len))
        return tuple(y.shape[1:])

    @property
    def n_frames(self) -> int:
        return self.shape[-1]

    @property
    def padding(self) -> tuple[int, int]:
        amount = self.n_fft - self.hop_length
        return floor(amount / 2), ceil(amount / 2)

    def _auto_pad(self, y: torch.Tensor) -> torch.Tensor:
        return F.pad(y, self.padding, mode="reflect")

    def _engine(self, y: torch.Tensor) -> torch.Tensor:
        B, *_ = y.shape
        spec = torch.stft(
            input=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.stft_window,
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = spec.abs().pow(2).add(self.eps)
        spec = torch.bmm(self.mel_basis[None, ...].repeat(B, 1, 1), spec)
        return compress_drange(spec)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self._engine(self._auto_pad(y))


if __name__ == "__main__":
    import librosa
    import matplotlib.pyplot as plt

    PLOT: bool = False
    SAMPLE_RATE = 22_050
    N_SEGMENTS = 64
    SEQ_LEN: int = 8192
    DURATION = (SEQ_LEN * N_SEGMENTS) / SAMPLE_RATE

    self = MelSpec(SEQ_LEN, sr=SAMPLE_RATE)

    y, _ = librosa.load(librosa.ex("sweetwaltz"), sr=self.sr, duration=DURATION)
    y = torch.from_numpy(y).unsqueeze(0)

    X = self.forward(y)

    print(X.shape)
    assert X.shape == (y.shape[0], self.n_mels, self.n_frames * N_SEGMENTS)

    if PLOT:
        plt.imshow(X[0], aspect="auto", cmap="turbo")
        plt.show()
