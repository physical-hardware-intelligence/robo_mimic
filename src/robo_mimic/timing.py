"""Per-stage timing. A single FPS number hides which stage is the problem.

Phase 6 made the case: the loop was running at 113 fps and looked fine, while
`project` was quietly burning 8.47 ms -- more than MediaPipe -- on IK calls for
a pan branch that won 0 of 120 frames. No aggregate number would have shown
that. The per-stage table did, immediately.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from types import TracebackType

import numpy as np


@dataclass(frozen=True, slots=True)
class StageStats:
    stage: str
    mean_ms: float
    p95_ms: float
    max_ms: float
    samples: int


@dataclass
class Budget:
    """Rolling per-stage timings in milliseconds, newest `window` kept.

    `stages` fixes the report order so the live readout and the printed table
    always agree, and so a stage that has not run yet still has a known place.
    """

    stages: tuple[str, ...]
    window: int = 60
    _samples: dict[str, deque[float]] = field(default_factory=dict, repr=False)

    def add(self, stage: str, milliseconds: float) -> None:
        if stage not in self.stages:
            raise KeyError(f"unknown stage {stage!r}; expected one of {self.stages}")
        self._samples.setdefault(stage, deque(maxlen=self.window)).append(milliseconds)

    def measure(self, stage: str) -> Clock:
        """`with budget.measure("project"): ...`"""
        return Clock(self, stage)

    def mean_ms(self, stage: str) -> float:
        values = self._samples.get(stage)
        return float(np.mean(values)) if values else 0.0

    def total_ms(self) -> float:
        """Sum of the per-stage means. The frame's budget."""
        return float(sum(self.mean_ms(stage) for stage in self.stages))

    def table(self) -> list[StageStats]:
        out = []
        for stage in self.stages:
            values = self._samples.get(stage)
            if not values:
                continue
            array = np.asarray(values, dtype=float)
            out.append(
                StageStats(
                    stage=stage,
                    mean_ms=float(array.mean()),
                    p95_ms=float(np.percentile(array, 95)),
                    max_ms=float(array.max()),
                    samples=len(array),
                )
            )
        return out

    def headroom_ms(self, rate_hz: float) -> float:
        """Spare time per frame at `rate_hz`. Negative means it does not fit."""
        return 1000.0 / rate_hz - self.total_ms()


class Clock:
    """Records one stage's duration. Records it even if the body raises."""

    __slots__ = ("_budget", "_stage", "_start")

    def __init__(self, budget: Budget, stage: str) -> None:
        self._budget = budget
        self._stage = stage
        self._start = 0.0

    def __enter__(self) -> Clock:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Recorded unconditionally: a stage that failed still consumed time, and
        # losing the sample would silently flatter the budget.
        self._budget.add(self._stage, (time.perf_counter() - self._start) * 1000.0)


__all__ = ["Budget", "Clock", "StageStats"]
