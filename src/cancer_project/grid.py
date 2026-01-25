"""
Minimal multicell emergence model (systems demo only).

Grid owns the system state.
Cells provide update dynamics only.

This is NOT a medical model.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

import numpy as np

# ------------------------------------------------------------
# Single-cell components
# ------------------------------------------------------------
try:
    from cancer_project import Environment, HealthyCell, CancerCell
except Exception as e:
    raise ImportError(
        "Missing Environment / HealthyCell / CancerCell. "
        "Check cancer_project/__init__.py"
    ) from e

# ------------------------------------------------------------
# Optional intervention API
# ------------------------------------------------------------
try:
    from cancer_project.intervention import (
        Intervention,
        InterventionContext,
        NoOpIntervention,
    )
except Exception:
    Intervention = None
    InterventionContext = None
    NoOpIntervention = None


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
@dataclass
class GridConfig:
    n: int = 20
    steps: int = 60
    init_cancer_prob: float = 0.03
    seed: Optional[int] = None

    neighbor_strength: float = 0.05
    mutation_prob: float = 0.001
    min_lambda: float = 0.0
    max_lambda: float = 2.0
    init_lambda: float = 1.0


# ------------------------------------------------------------
# Grid model
# ------------------------------------------------------------
class Grid:
    def __init__(
        self,
        cfg: GridConfig,
        intervention: Optional["Intervention"] = None,
    ):
        self.cfg = cfg
        self.n = cfg.n
        self.t = 0

        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

        self.env = Environment()

        # Cells
        self.cells = np.empty((self.n, self.n), dtype=object)

        # System state (owned by grid)
        self.gv = np.zeros((self.n, self.n), dtype=float)
        self.lam = np.full((self.n, self.n), cfg.init_lambda, dtype=float)

        for i in range(self.n):
            for j in range(self.n):
                if random.random() < cfg.init_cancer_prob:
                    self.cells[i, j] = CancerCell(self.env)
                else:
                    self.cells[i, j] = HealthyCell(self.env)

        if intervention is None and NoOpIntervention is not None:
            self.intervention = NoOpIntervention()
        else:
            self.intervention = intervention

    # --------------------------------------------------------
    # Neighborhood
    # --------------------------------------------------------
    def neighbors(self, i: int, j: int):
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.n and 0 <= nj < self.n:
                yield ni, nj

    # --------------------------------------------------------
    # Simulation step
    # --------------------------------------------------------
    def step(self):
        # Cell intrinsic dynamics
        for i in range(self.n):
            for j in range(self.n):
                self.cells[i, j].step(self.env)
                self.gv[i, j] += self.env.gv  # accumulate strain signal

        # Local coupling
        for i in range(self.n):
            for j in range(self.n):
                for ni, nj in self.neighbors(i, j):
                    if isinstance(self.cells[i, j], HealthyCell):
                        self.lam[ni, nj] += self.cfg.neighbor_strength
                    else:
                        self.lam[ni, nj] -= self.cfg.neighbor_strength

        # Mutation / erosion
        for i in range(self.n):
            for j in range(self.n):
                if random.random() < self.cfg.mutation_prob:
                    self.lam[i, j] *= 0.5

                self.lam[i, j] = float(
                    np.clip(
                        self.lam[i, j],
                        self.cfg.min_lambda,
                        self.cfg.max_lambda,
                    )
                )

        # Intervention hook
        if self.intervention and InterventionContext:
            ctx = InterventionContext(t=self.t)
            self.intervention.apply(self, ctx)

        self.t += 1

    # --------------------------------------------------------
    # Fields (for plotting)
    # --------------------------------------------------------
    def gv_field(self) -> np.ndarray:
        return self.gv.copy()

    def lambda_field(self) -> np.ndarray:
        return self.lam.copy()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    def mean_gv(self) -> float:
        return float(np.mean(self.gv))

    def mean_lambda(self) -> float:
        return float(np.mean(self.lam))


# ------------------------------------------------------------
# CLI entrypoint
# ------------------------------------------------------------
def run_demo():
    cfg = GridConfig()
    grid = Grid(cfg)

    print("t,mean_gv,mean_lambda")
    for _ in range(cfg.steps):
        grid.step()
        print(f"{grid.t},{grid.mean_gv():.6f},{grid.mean_lambda():.6f}")


if __name__ == "__main__":
    run_demo()
