"""

    Utils

"""
from __future__ import annotations

import inspect
import time
from math import ceil, log2
from typing import Any, Callable, Dict, Optional, Tuple

import git
import torch


def remove_ext(i: str) -> str:
    return i.split(".", 1)[0]


def next_pow2(i: int) -> int:
    return 2 ** ceil(log2(i))


def is_pow2(i: int) -> int:
    return log2(i) % 1 == 0


def tiny(tensor: torch.Tensor) -> float:
    # Machine epsilon for `tensor`s dtype
    return torch.finfo(tensor.dtype).eps


def get_git_metadata() -> Optional[Dict[str, str]]:
    try:
        repo = git.Repo(search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return None

    return dict(
        sha_short=repo.head.object.hexsha[:8],
        sha_long=repo.head.object.hexsha,
        branch=repo.active_branch.name,
    )


class Timer:
    def __init__(
        self,
        name: Optional[str] = None,
        ndigits: int = 5,
        print_on_enter: bool = True,
    ) -> None:
        self.name = name
        self.ndigits = ndigits
        self.print_on_enter = print_on_enter

        self.start_time: Optional[float] = None

    def __enter__(self) -> None:
        if self.print_on_enter:
            print(f"Timer starting [name={self.name}]...")
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_tb:
            delta = round(time.time() - self.start_time, self.ndigits)
            print(f"Elapsed Time [name={self.name}]: {delta}")
