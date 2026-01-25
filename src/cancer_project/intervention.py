"""
Minimal constraint-restoration interventions.

Systems demo only.
No biology. No medicine.
"""

from __future__ import annotations
from dataclasses import dataclass
import random


@dataclass
class LocalLambdaRepair:
    """
    Local intervention that restores constraint tightness (lambda)
    when GV strain exceeds a threshold.

    This models recoverability of feedback, not drugs.
    """
    gv_threshold: float = 0.6
    repair_amount: float = 0.05
    max_lambda: float = 2.0
    probability: float = 0.3  # stochastic, immune-like

    def apply(self, grid, t: int) -> None:
        for i in range(grid.n):
            for j in range(grid.n):
                cell = grid.cells[i][j]

                # Only act in high-strain regions
                if cell.gv > self.gv_threshold:
                    if random.random() < self.probability:
                        cell.lam = min(
                            self.max_lambda,
                            cell.lam + self.repair_amount
                        )
