"""
Minimal multicell emergence model (systems demo only).

A 2D grid of sites. Each site has:
- gv: accumulated "strain / risk" scalar
- lam: local constraint tightness (feedback strength)

This is NOT a medical model.
It is a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

import numpy as np

# ------------------------------------------------------------
# Import single-cell components
# ------------------------------------------------------------
try:
    from cancer_project import Environment, HealthyCell, CancerCell
except Exception as e:
    raise ImportError(
        "Could not import Environment / HealthyCell / CancerCell. "
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
    Intervention = None  # type: ignore
    InterventionContext = None  # type: ignore
    NoOpIntervention = None  # type: ignore


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

        self.cells = np.empty((self.n, self.n), dtype=object)
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
        # Intrinsic updates
        for i in range(self.n):
            for j in range(self.n):
                self.cells[i, j].step(self.env)

        # Local coupling
        for i in range(self.n):
            for j in range(self.n):
                cell = self.cells[i, j]
                for ni, nj in self.neighbors(i, j):
                    neighbor = self.cells[ni, nj]
                    if isinstance(cell, HealthyCell):
                        neighbor.lam += self.cfg.neighbor_strength
                    else:
                        neighbor.lam -= self.cfg.neighbor_strength

        # Mutation / constraint erosion
        for i in range(self.n):
            for j in range(self.n):
                if random.random() < self.cfg.mutation_prob:
                    self.cells[i, j].lam *= 0.5

                self.cells[i, j].lam = float(
                    np.clip(
                        self.cells[i, j].lam,
                        self.cfg.min_lambda,
                        self.cfg.max_lambda,
                    )
                )

        # Intervention hook
        if self.intervention is not None and InterventionContext is not None:
            ctx = InterventionContext(t=self.t)
            self.intervention.apply(self, ctx)

        self.t += 1

    # --------------------------------------------------------
    # Fields (for plotting)
    # --------------------------------------------------------
    def gv_field(self) -> np.ndarray:
        return np.array([[c.gv for c in row] for row in self.cells])

    def lambda_field(self) -> np.ndarray:
        return np.array([[c.lam for c in row] for row in self.cells])

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    def mean_gv(self) -> float:
        return float(np.mean(self.gv_field()))

    def mean_lambda(self) -> float:
        return float(np.mean(self.lambda_field()))


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
