"""
Minimal multicell emergence model (systems demo only).

A 2D grid of sites. Each site has:
- gv: accumulated "strain / risk" scalar
- lam: local constraint tightness (feedback strength)

Local coupling:
- Healthy sites reinforce neighbor lam
- Cancer sites erode neighbor lam
- Rare stochastic "mutation" events reduce lam

This is NOT a medical model. It is a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple, Optional

import numpy as np

# Import single-cell components (must exist in cancer_project)
from cancer_project import Environment, HealthyCell, CancerCell


# ============================================================
# Configuration
# ============================================================

@dataclass
class GridConfig:
    # geometry / time
    n: int = 20
    steps: int = 60
    seed: int = 7

    # initialization
    init_cancer_prob: float = 0.03

    # constraint field (lambda)
    lam_init: float = 2.0
    lam_min: float = 0.0
    lam_max: float = 2.0

    # coupling
    healthy_repair: float = 0.01
    cancer_erosion: float = 0.02
    mutation_prob: float = 0.002
    mutation_drop: float = 0.3


# ============================================================
# Grid
# ============================================================

class Grid:
    def __init__(self, cfg: GridConfig, env: Environment):
        self.cfg = cfg
        self.env = env
        self.n = cfg.n

        rng = random.Random(cfg.seed)

        self.cells: List[List[HealthyCell | CancerCell]] = []

        for i in range(self.n):
            row = []
            for j in range(self.n):
                if rng.random() < cfg.init_cancer_prob:
                    cell = CancerCell()
                else:
                    cell = HealthyCell()

                # attach constraint field
                cell.lam = cfg.lam_init
                row.append(cell)
            self.cells.append(row)

        self.t = 0

    # --------------------------------------------------------

    def neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        out = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.n and 0 <= nj < self.n:
                out.append((ni, nj))
        return out

    # --------------------------------------------------------

    def step(self):
        """Advance one timestep."""

        rng = random.Random(self.cfg.seed + self.t)

        # --- cell internal dynamics ---
        for i in range(self.n):
            for j in range(self.n):
                self.cells[i][j].step(self.env, rng=rng)

        # --- constraint coupling ---
        for i in range(self.n):
            for j in range(self.n):
                cell = self.cells[i][j]

                if isinstance(cell, HealthyCell):
                    delta = self.cfg.healthy_repair
                else:
                    delta = -self.cfg.cancer_erosion

                for ni, nj in self.neighbors(i, j):
                    ncell = self.cells[ni][nj]
                    ncell.lam += delta

        # --- stochastic mutation ---
        for i in range(self.n):
            for j in range(self.n):
                if rng.random() < self.cfg.mutation_prob:
                    self.cells[i][j].lam -= self.cfg.mutation_drop

        # --- clamp lam ---
        for i in range(self.n):
            for j in range(self.n):
                cell = self.cells[i][j]
                cell.lam = max(self.cfg.lam_min, min(self.cfg.lam_max, cell.lam))

        self.t += 1

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def gv_field(self) -> np.ndarray:
        return np.array([[cell.gv for cell in row] for row in self.cells])

    def lam_field(self) -> np.ndarray:
        return np.array([[cell.lam for cell in row] for row in self.cells])

    def mean_gv(self) -> float:
        return float(np.mean(self.gv_field()))

    def mean_lam(self) -> float:
        return float(np.mean(self.lam_field()))


# ============================================================
# Demo entrypoint
# ============================================================

def run_demo():
    cfg = GridConfig()
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)
    grid = Grid(cfg, env)

    print("t,mean_gv,mean_lambda")
    for t in range(cfg.steps):
        grid.step()
        print(f"{t},{grid.mean_gv():.6f},{grid.mean_lam():.6f}")


# backwards compatibility
run_grid = run_demo


if __name__ == "__main__":
    run_demo()
