"""
Intervention API (systems demo only).

Defines a minimal, stable interface for constraint-restoration
interventions that act on the Grid without hard-coded biology.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class InterventionContext:
    t: int  # current timestep


class Intervention:
    """
    Base protocol for interventions.
    """
    def apply(self, grid: "Grid", ctx: InterventionContext) -> None:
        raise NotImplementedError


class NoOpIntervention(Intervention):
    def apply(self, grid: "Grid", ctx: InterventionContext) -> None:
        return


class LambdaRestorationPulse(Intervention):
    """
    Simple systems-level "therapy":

    - Detects high-GV sites
    - Locally restores lambda (constraint strength)
    - No biology, no targeting logic beyond strain
    """

    def __init__(
        self,
        gv_threshold: float = 0.05,
        restore_amount: float = 0.15,
        every: int = 5,
    ):
        self.gv_threshold = gv_threshold
        self.restore_amount = restore_amount
        self.every = every

    def apply(self, grid: "Grid", ctx: InterventionContext) -> None:
        # Only pulse every N steps
        if ctx.t % self.every != 0:
            return

        gv = grid.gv
        lam = grid.lam
        cfg = grid.cfg

        mask = gv > self.gv_threshold
        lam[mask] += self.restore_amount

        # clamp
        np.clip(lam, cfg.lambda_min, cfg.lambda_max, out=lam)
