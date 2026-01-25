"""
Intervention API (systems demo only).

Goal:
- Define a minimal, stable interface for "therapies" that restore
  constraint integrity (lambda) without encoding biology.

This is NOT a medical model.
It is a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class InterventionContext:
    """
    Read-only metadata available to interventions.

    Keep this minimal so the API remains stable as the model evolves.
    """
    t: int  # current timestep


class Intervention(Protocol):
    """
    An intervention modifies the grid *in place*.
    """

    def apply(self, grid: "Grid", ctx: InterventionContext) -> None:
        ...


class LocalConstraintRestoration:
    """
    Minimal intervention:
    - If a cell's GV exceeds a threshold,
      locally restore constraint tightness (lambda).

    This tests whether invasion fronts are reversible
    via feedback recovery alone.
    """

    def __init__(
        self,
        gv_threshold: float = 0.5,
        restore_strength: float = 0.05,
        max_lambda: float = 1.5,
    ):
        self.gv_threshold = gv_threshold
        self.restore_strength = restore_strength
        self.max_lambda = max_lambda

    def apply(self, grid: "Grid", ctx: InterventionContext) -> None:
        for i in range(grid.n):
            for j in range(grid.n):
                cell = grid.cells[i][j]

                if cell.gv > self.gv_threshold:
                    cell.lambda_ = min(
                        self.max_lambda,
                        cell.lambda_ + self.restore_strength,
                    )
