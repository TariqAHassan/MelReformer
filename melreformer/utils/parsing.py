"""

    Parsing

"""
from __future__ import annotations

import torch


def to_numeric(value: int | float | torch.Tensor) -> int | float:
    if isinstance(value, (int, float)):
        return value
    elif isinstance(value, torch.Tensor):
        return value.item()
    else:
        raise TypeError(f"Got unexpected value type '{type(value)}'")
