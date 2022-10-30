"""

    Metrics

"""
from __future__ import annotations

from collections import defaultdict, deque
from statistics import mean, stdev
from typing import Any, Callable, DefaultDict, Deque, Dict, List, Optional, Union

import torch


def _none_double(value: Any) -> Any:
    if value is None:
        return None, None
    else:
        return value


class MetricTracker:
    memory: DefaultDict[str, Union[Deque, List[Any]]]

    def __init__(self, n: Optional[int] = 100) -> None:
        self.n = n

        if n is None:
            self.memory = defaultdict(list)
        else:
            self.memory = defaultdict(lambda: deque(maxlen=n))

    def reset(self) -> MetricTracker:
        for v in self.memory.values():
            v.clear()
        return self

    @staticmethod
    def _normalize(value: Union[int, float, torch.Tensor]) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.item())
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            raise TypeError(f"Unable to normalize value of type {value}")

    def update(
        self,
        *values: Union[int, float, torch.Tensor],
        name: Optional[str] = None,
    ) -> MetricTracker:
        self.memory[name].append(mean(map(self._normalize, values)))
        return self

    def _compute_stat(
        self,
        calculator: Callable[[Deque | list[Any]], Union[int, float]],
        min_size: int,
        name: Optional[str],
    ) -> Optional[Union[int, float]]:
        if name in self.memory:
            data = self.memory[name]
            if len(data) >= min_size:
                return calculator(data)
            else:
                return None
        else:
            raise KeyError(f"No metric '{name}'")

    def count(self, name: Optional[str] = None) -> Optional[int]:
        return self._compute_stat(len, min_size=0, name=name)

    def count_all(self) -> Dict[Optional[str], Optional[float]]:
        return {k: self.count(k) for k in self.memory}

    def avg(self, name: Optional[str] = None) -> Optional[float]:
        return self._compute_stat(mean, min_size=1, name=name)

    def avg_all(self) -> Dict[Optional[str], Optional[float]]:
        return {k: self.avg(k) for k in self.memory}

    def stdev(self, name: Optional[str] = None) -> Optional[float]:
        return self._compute_stat(stdev, min_size=2, name=name)

    def stdev_all(self) -> Dict[Optional[str], Optional[float]]:
        return {k: self.stdev(k) for k in self.memory}


class BestMetricTracker:
    def __init__(self, metrics: dict[str, str]) -> None:
        self.metrics = metrics

        for k, v in metrics.items():
            if v not in ("min", "max"):
                raise ValueError(f"Got invalid direction for '{k}'")

        self.best = dict.fromkeys(self.metrics)

    def __contains__(self, item: str) -> bool:
        return item in self.metrics

    def update(self, name: str, value: int | float, current_step: int) -> bool:
        last_value, last_step = _none_double(self.best[name])
        if last_step is not None and last_step == current_step:
            return False
        elif (
            last_value is None
            or (self.metrics[name] == "max" and value > last_value)
            or (self.metrics[name] == "min" and value < last_value)
        ):
            self.best[name] = (value, current_step)
            return True
        return False


if __name__ == "__main__":
    btracker = BestMetricTracker(dict(fid="min", score="max"))

    assert "fid" in btracker
    assert not "loss" in btracker

    # Check update succeeds
    assert btracker.update("fid", value=10, current_step=0)
    assert btracker.update("score", value=10, current_step=0)

    # Check repeated update (with same step) fails
    assert not btracker.update("fid", value=10, current_step=0)
    assert not btracker.update("score", value=10, current_step=0)

    # Check FID update fails b/c current FID is > last FID
    assert not btracker.update("fid", value=11, current_step=1)

    # Check score update succeeds b/c current score > last score
    assert btracker.update("score", value=11, current_step=1)
