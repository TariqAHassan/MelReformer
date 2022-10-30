"""

    Multi-Period Discriminator

    References:
        * https://arxiv.org/abs/2010.05646
        * https://github.com/jik876/hifi-gan

"""
from functools import cached_property
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class PeriodDiscriminator(nn.Module):
    def __init__(
        self,
        period: int = 2,
        seq_len: int = 2**16,
        input_channels: int = 1,
        channel_seq: tuple[int, ...] = (32, 128, 512, 1024, 1024),
        kernel_size: int = 5,
        stride: int = 3,
    ) -> None:
        super().__init__()
        self.period = period
        self.seq_len = seq_len
        self.input_channels = input_channels
        self.channel_seq = channel_seq
        self.kernel_size = kernel_size
        self.stride = stride

        self.blocks = nn.ModuleList([])
        for i in range(1, len(self.channels)):
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels[i - 1],
                    out_channels=self.channels[i],
                    kernel_size=(kernel_size, 1),
                    stride=(1 if i == len(self.channels) - 1 else stride, 1),
                    padding=(_get_padding(kernel_size, dilation=1), 0),
                ),
                nn.LeakyReLU(0.2),
            )
            self.blocks.append(block)

        self.conv_post = nn.Conv2d(
            in_channels=self.blocks[-1][0].out_channels,
            out_channels=1,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )

    @property
    def channels(self) -> tuple[int, ...]:
        return (self.input_channels, *self.channel_seq)

    @cached_property
    def embedding_size(self) -> int:
        return self.forward(torch.randn(1, self.seq_len))[0].shape[-1]

    def _1d_to_2d(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), mode="reflect")
            t = t + n_pad
        return x.view(b, c, t // self.period, self.period)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = list()
        y = self._1d_to_2d(x.unsqueeze(1) if x.ndim == 2 else x)
        for block in self.blocks:
            y = block(y)
            fmap.append(y)
        y = self.conv_post(y)
        fmap.append(y)
        return y.flatten(1), fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(
        self,
        periods: tuple[int, ...] = (2, 3, 5, 7, 11),
        seq_len: int = 2**16,
        input_channels: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.periods = periods
        self.seq_len = seq_len
        self.input_channels = input_channels

        self.discriminators = nn.ModuleList(
            [
                PeriodDiscriminator(
                    period=p,
                    seq_len=seq_len,
                    input_channels=input_channels,
                    **kwargs,
                )
                for p in periods
            ]
        )

    @cached_property
    def embedding_size(self) -> int:
        ys, _ = self.forward(torch.randn(1, self.input_channels, self.seq_len))
        return sum(i.numel() for i in ys)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        ys, fmaps = list(), list()
        for d in self.discriminators:
            y, fmap = d(x)
            ys.append(y)
            fmaps.append(fmap)
        return ys, fmaps


if __name__ == "__main__":
    from melreformer.utils.general import Timer
    from melreformer.utils.training import count_parameters

    self = mpd = MultiPeriodDiscriminator()
    print(f"MPD(x) Params: {count_parameters(mpd):,}")

    x = torch.randn(1, 2**16)
    with Timer("MultiPeriodDiscriminator"):
        ys, fmaps = mpd(x)

    assert [i.shape for i in ys] == [
        torch.Size([1, 810]),
        torch.Size([1, 810]),
        torch.Size([1, 810]),
        torch.Size([1, 812]),
        torch.Size([1, 814]),
    ]
    assert all(all(isinstance(i, torch.Tensor) for i in fmap) for fmap in fmaps)
