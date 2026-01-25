"""
Minimal multicell emergence model (systems demo only).

A 2D grid of cells. Each site has:
- gv: accumulated strain / risk scalar
- lam: local constraint tightness (feedback strength)

Healthy cells reinforce neighbor lam.
Cancer cells erode neighbor lam.

This is NOT a medical model.
It is a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple
import random
import numpy as np

# Import single-cell components
from cancer_project import Environment, HealthyCell, CancerCell


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class GridConfig:
    n: int = 20
    steps: int = 60
    init_cancer_prob: float = 0.03
    seed: Optional[int] = None


# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------

class Grid:
    def __init__(self, cfg: GridConfig, env: Optional[Environment] = None):
        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

        self.cfg = cfg
        self.n = cfg.n
        self.t = 0

        # Environment (accept injected OR create)
        self.env = env if env is not None else Environment()

        # Allocate fields
        self.cells: List[List[object]] = []
        self.gv = np.zeros((self.n, self.n), dtype=float)
        self.lam = np.ones((self.n, self.n), dtype=float)

        # Initialize grid
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if random.random() < cfg.init_cancer_prob:
                    cell = CancerCell()
                else:
                    cell = HealthyCell()
                row.append(cell)
            self.cells.append(row)

    # -----------------------------------------------------------------
    # Neighborhood helpers
    # -----------------------------------------------------------------

    def neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        out = []
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.n and 0 <= nj < self.n:
                out.append((ni, nj))
        return out

    # -----------------------------------------------------------------
    # One timestep
    # -----------------------------------------------------------------

    def step(self, dt: float = 1.0):
        self.t += 1

        # 1) Cell internal dynamics
        for i in range(self.n):
            for j in range(self.n):
                self.cells[i][j].step(self.env, dt)

        # 2) Constraint coupling (lambda)
        new_lam = self.lam.copy()

        for i in range(self.n):
            for j in range(self.n):
                cell = self.cells[i][j]
                for ni, nj in self.neighbors(i, j):
                    if isinstance(cell, HealthyCell):
                        new_lam[ni, nj] += 0.02
                    else:
                        new_lam[ni, nj] -= 0.04

        self.lam = np.clip(new_lam, 0.0, 2.0)

        # 3) GV accumulation
        for i in range(self.n):
            for j in range(self.n):
                if isinstance(self.cells[i][j], CancerCell):
                    self.gv[i, j] += 0.05 * (1.0 + (1.0 - self.lam[i, j]))
                else:
                    self.gv[i, j] *= 0.97

    # -----------------------------------------------------------------
    # Field accessors (plot-safe)
    # -----------------------------------------------------------------

    def gv_field(self) -> np.ndarray:
        return self.gv.copy()

    def lam_field(self) -> np.ndarray:
        return self.lam.copy()

    # Backward compatibility
    def lambda_field(self) -> np.ndarray:
        return self.lam_field()

    # -----------------------------------------------------------------
    # Run loop
    # -----------------------------------------------------------------

    def run(self):
        for _ in range(self.cfg.steps):
            self.step()

    # -----------------------------------------------------------------
    # CLI entry
    # -----------------------------------------------------------------

    @staticmethod
    def main():
        cfg = GridConfig()
        grid = Grid(cfg)
        grid.run()

        for t in range(cfg.steps):
            print(f"{t},{grid.gv.mean():.6f},{grid.lam.mean():.6f}")


if __name__ == "__main__":
    Grid.main()
