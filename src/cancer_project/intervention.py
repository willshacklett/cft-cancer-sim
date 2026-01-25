from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Protocol
import math
import random


@dataclass
class InterventionConfig:
    """
    Intervention knobs (kept intentionally non-biological).

    Units:
      - lambda_boost: additive boost to lambda per application
      - gv_damp: multiplicative damping applied to GV (e.g. 0.98 means 2% reduction)
    """
    enabled: bool = True

    # Core effects
    lambda_boost: float = 0.02
    gv_damp: float = 1.0  # 1.0 = no damping, 0.98 = reduce GV 2%

    # Where/how to apply
    radius: int = 2              # neighborhood radius for local effects
    diffusion_decay: float = 0.6 # exp decay per Manhattan distance (0<decay<=1)

    # Triggering
    gv_threshold: float = 0.25   # activate when local GV exceeds this
    budget_per_step: int = 5     # max number of targeted centers per step

    # Cost / trade-off (to avoid trivial solutions)
    energy_cost: float = 0.0     # increases GV slightly when you treat (cost)
    overconstraint_lambda: float = 1.5  # if lambda exceeds this, apply "rigidity penalty"
    rigidity_cost: float = 0.01  # adds GV if lambda too high


class InterventionPolicy(Protocol):
    """
    Minimal policy interface: apply intervention to grid at timestep t.
    """
    cfg: InterventionConfig

    def apply(self, grid, t: int) -> None:
        ...


class NoIntervention:
    def __init__(self, cfg: Optional[InterventionConfig] = None):
        self.cfg = cfg or InterventionConfig(enabled=False)

    def apply(self, grid, t: int) -> None:
        return


class GlobalPulse:
    """
    Applies a uniform repair pulse at fixed cadence.
    """
    def __init__(self, cfg: InterventionConfig, every: int = 10, start: int = 0):
        self.cfg = cfg
        self.every = max(1, every)
        self.start = start

    def apply(self, grid, t: int) -> None:
        if not self.cfg.enabled:
            return
        if t < self.start or (t - self.start) % self.every != 0:
            return

        for cell in grid.cells_flat():
            cell.lambda_ = min(cell.lambda_ + self.cfg.lambda_boost, self.cfg.overconstraint_lambda * 2)
            cell.gv *= self.cfg.gv_damp
            # treatment cost
            cell.gv += self.cfg.energy_cost
            # rigidity penalty
            if cell.lambda_ > self.cfg.overconstraint_lambda:
                cell.gv += self.cfg.rigidity_cost


class LocalDiffuse:
    """
    A fixed set of 'sources' emits a restorative field that decays with distance.
    """
    def __init__(self, cfg: InterventionConfig, sources: Tuple[Tuple[int, int], ...]):
        self.cfg = cfg
        self.sources = sources

    def apply(self, grid, t: int) -> None:
        if not self.cfg.enabled:
            return

        for (sx, sy) in self.sources:
            self._apply_source(grid, sx, sy)

    def _apply_source(self, grid, sx: int, sy: int) -> None:
        r = self.cfg.radius
        for x in range(max(0, sx - r), min(grid.w, sx + r + 1)):
            for y in range(max(0, sy - r), min(grid.h, sy + r + 1)):
                d = abs(x - sx) + abs(y - sy)
                strength = (self.cfg.diffusion_decay ** d)
                if strength <= 0:
                    continue
                cell = grid.cell(x, y)
                cell.lambda_ = min(cell.lambda_ + self.cfg.lambda_boost * strength,
                                   self.cfg.overconstraint_lambda * 2)
                cell.gv *= (1.0 - (1.0 - self.cfg.gv_damp) * strength)
                cell.gv += self.cfg.energy_cost * strength
                if cell.lambda_ > self.cfg.overconstraint_lambda:
                    cell.gv += self.cfg.rigidity_cost * strength


class ThresholdTargeted:
    """
    Each timestep: find high-GV cells, target up to budget_per_step centers,
    and apply a local diffuse pulse around each center.

    This is the most "immune-like" without calling it immune.
    """
    def __init__(self, cfg: InterventionConfig):
        self.cfg = cfg

    def apply(self, grid, t: int) -> None:
        if not self.cfg.enabled:
            return

        # Collect candidates above threshold
        candidates = []
        for x in range(grid.w):
            for y in range(grid.h):
                c = grid.cell(x, y)
                if c.gv >= self.cfg.gv_threshold:
                    candidates.append((c.gv, x, y))

        if not candidates:
            return

        # Sort descending by GV (highest strain gets treated)
        candidates.sort(reverse=True, key=lambda z: z[0])

        # Pick up to budget centers
        centers = [(x, y) for _, x, y in candidates[: self.cfg.budget_per_step]]

        # Apply local diffuse around each center
        ld = LocalDiffuse(self.cfg, tuple(centers))
        ld.apply(grid, t)
