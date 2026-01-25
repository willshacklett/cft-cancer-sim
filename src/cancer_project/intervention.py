"""
Intervention API (systems demo only).

Goal:
- Define a clean interface for "therapies" that can modify grid state
  (GV, lambda, or environment signals) without hard-coding biology.

This is not a medical model. It's a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class InterventionContext:
    """
    Read-only metadata you can use to make intervention decisions.

    Keep this minimal so the API stays stable as the model evolves.
    """
    t: int  # current timestep


class Intervention(Protocol):
    """
    Implementations can modify the grid in-place.
    """

    def apply(self, grid: "Grid", ctx: InterventionContext) -> None:
        """
        Apply an intervention step to the grid.
        """
        ...


@dataclass
class NoOpIntervention:
    """
    Default intervention: does nothing.
    """

    def apply(self, grid: "Grid", ctx: InterventionContext) -> None:
        return


# Optional helper: a simple "pulse" schedule you can reuse later
@dataclass
class PulseSchedule:
    start: int = 0
    end: Optional[int] = None
    every: int = 1  # apply every N steps

    def active(self, t: int) -> bool:
        if t < self.start:
            return False
        if self.end is not None and t > self.end:
            return False
        return (t - self.start) % max(1, self.every) == 0
