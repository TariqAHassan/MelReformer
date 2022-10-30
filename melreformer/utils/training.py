"""

    Training

"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.utils.data import Dataset, random_split


def count_parameters(model: nn.Module, grad_req: bool = True) -> int:
    # See https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    return sum(p.numel() for p in model.parameters() if p.requires_grad is grad_req)


def toggle_grad(model: nn.Module, requires_grad: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def train_val_split(
    dataset: Dataset,
    val_amount: Optional[Union[int, float]],
    val_ceiling: Optional[int] = None,
    seed: int = 42,
) -> Tuple[Dataset, Optional[Dataset]]:
    if val_amount is None:
        return Dataset, None

    if isinstance(val_amount, float):
        n_val = int(len(dataset) * val_amount)
    elif isinstance(val_amount, int):
        n_val = val_amount
    else:
        raise TypeError(f"Unable to handle val_amount {val_amount}")

    if val_ceiling and n_val > val_ceiling:
        n_val = val_ceiling
    return random_split(
        dataset=dataset,
        lengths=[len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


def update_average(
    model_tgt: nn.Module,
    model_src: nn.Module,
    beta: Optional[Union[int, float]],
) -> None:
    toggle_grad(model_tgt, requires_grad=False)
    toggle_grad(model_src, requires_grad=False)

    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        if p_src is p_tgt:
            raise ValueError("p_src == p_tgt")

        if beta is None:
            p_tgt.copy_(p_src)
        else:
            p_tgt.copy_(beta * p_tgt + (1.0 - beta) * p_src)

    toggle_grad(model_tgt, requires_grad=True)
    toggle_grad(model_src, requires_grad=True)


def compute_total_norm(
    parameters: Iterable[torch.Tensor],
    norm_type: float = 2.0,
) -> torch.Tensor:
    # From torch.nn.utils.clip_grad.clip_grad_norm_.
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    norm_type = float(norm_type)
    parameters = [p for p in parameters if p.grad is not None]

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device
    if norm_type == math.inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            p=norm_type,
        )
    return total_norm
