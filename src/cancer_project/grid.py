"""
Minimal multicell emergence model (systems demo only).

A 2D grid of cells where each site has:
- GV: accumulated "strain/risk" scalar
- lambda: local constraint tightness (feedback strength)

Local coupling:
- Healthy cells reinforce neighbor lambda
- Cancer cells erode neighbor lambda
- Rare stochastic events reduce lambda ("mutation" as constraint relaxation)

This is not a medical model. It's a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple, Optional

import numpy as np

# We reuse your existing single-cell environment + cell classes if present.
# If your package exports these, this import will work.
try:
    from cancer_project import Environment, HealthyCell, CancerCell
except Exception as e:
    raise ImportError(
        "Could not import Environment/HealthyCell/CancerCell from cancer_project. "
        "Make sure those exist and are exported in cancer_project/__init__.py."
    ) from e


@dataclass
class GridConfig:
    n: int = 20
    steps: int = 60

    # initial state
    init_cancer_prob: float = 0.03

    # lambda dynamics
    lambda_init: float = 1.0
    lambda_min: float = 0.05
    lambda_max: float = 2.0

    # coupling strengths
    neighbor_strength: float = 0.010  # healthy -> increase neighbor lambda
    erosion_strength: float = 0.020   # cancer  -> decrease neighbor lambda

    # stochastic constraint relaxation
    mutation_prob: float = 0.002
    mutation_drop: float = 0.15  # subtract from lambda

    # GV update parameters (kept simple + interpretable)
    constraint_strength: float = 0.25
    strain_atp_weight: float = 0.35
    strain_damage_weight: float = 0.65
    strain_div_weight: float = 0.06
    strain_arrest_weight: float = 0.12

    seed: Optional[int] = 7


class Grid:
    def __init__(self, cfg: GridConfig, env: Environment):
        self.cfg = cfg
        self.env = env

        if cfg.seed is not None:
            random.seed(cfg.seed)
            np.random.seed(cfg.seed)

        self.n = int(cfg.n)

        # Cell objects
        self.cells: List[List[object]] = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if random.random() < cfg.init_cancer_prob:
                    row.append(CancerCell())
                else:
                    row.append(HealthyCell())
            self.cells.append(row)

        # Local constraint tightness field (lambda)
        self.lmbda = np.full((self.n, self.n), float(cfg.lambda_init), dtype=float)

        # GV scalar field
        self.gv = np.zeros((self.n, self.n), dtype=float)

        self.t = 0

    def _neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        out = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.n and 0 <= nj < self.n:
                out.append((ni, nj))
        return out

    def _is_cancer(self, cell: object) -> bool:
        return cell.__class__.__name__.lower().startswith("cancer")

    def _cell_strain(self, cell: object) -> float:
        """
        Simple strain proxy derived from existing single-cell state.
        We intentionally keep this minimal and robust.
        """
        atp = float(getattr(cell, "atp", 1.0))
        damage = float(getattr(cell, "damage", 0.0))
        arrest_steps = float(getattr(cell, "arrest_steps", 0.0))
        divisions = float(getattr(cell, "divisions", 0.0))

        strain = (
            self.cfg.strain_atp_weight * max(0.0, 1.0 - atp) +
            self.cfg.strain_damage_weight * max(0.0, damage) +
            self.cfg.strain_div_weight * max(0.0, divisions) +
            self.cfg.strain_arrest_weight * (1.0 if arrest_steps > 0 else 0.0)
        )
        return float(strain)

    def step(self) -> None:
        """
        One timestep:
        1) Step each cell with the shared environment
        2) Update lambda via local coupling + mutation
        3) Update GV using strain accumulation minus lambda feedback
        """
        n = self.n

        # 1) Step cells
        for i in range(n):
            for j in range(n):
                self.cells[i][j].step(self.env)

        # 2) Lambda coupling
        delta = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                cell = self.cells[i][j]
                if self._is_cancer(cell):
                    # cancer erodes neighbors
                    for ni, nj in self._neighbors(i, j):
                        delta[ni, nj] -= self.cfg.erosion_strength
                else:
                    # healthy reinforces neighbors
                    for ni, nj in self._neighbors(i, j):
                        delta[ni, nj] += self.cfg.neighbor_strength

        # apply stochastic constraint relaxation
        if self.cfg.mutation_prob > 0:
            mut_mask = np.random.rand(n, n) < self.cfg.mutation_prob
            delta[mut_mask] -= self.cfg.mutation_drop

        self.lmbda = np.clip(self.lmbda + delta, self.cfg.lambda_min, self.cfg.lambda_max)

        # 3) GV update
        for i in range(n):
            for j in range(n):
                cell = self.cells[i][j]
                strain = self._cell_strain(cell)

                # GV_{t+1} = GV_t + strain - lambda * constraint_strength * GV_t
                gv_t = self.gv[i, j]
                gv_next = gv_t + strain - (self.lmbda[i, j] * self.cfg.constraint_strength * gv_t)
                self.gv[i, j] = max(0.0, gv_next)

        self.t += 1

    def mean_gv(self) -> float:
        return float(self.gv.mean())

    def mean_lambda(self) -> float:
        return float(self.lmbda.mean())

    def gv_field(self) -> np.ndarray:
        return self.gv.copy()

    def lambda_field(self) -> np.ndarray:
        return self.lmbda.copy()


def run_grid(cfg: GridConfig | None = None, env: Environment | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the grid and returns (mean_gv_over_time, mean_lambda_over_time).
    """
    cfg = cfg or GridConfig()
    env = env or Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)

    grid = Grid(cfg, env)
    gv_hist = []
    lam_hist = []

    for _ in range(cfg.steps):
        grid.step()
        gv_hist.append(grid.mean_gv())
        lam_hist.append(grid.mean_lambda())

    return np.array(gv_hist, dtype=float), np.array(lam_hist, dtype=float)


def run_demo() -> None:
    """
    CLI/demo output: prints a simple emergence signal as CSV:
    t,mean_gv,mean_lambda
    """
    cfg = GridConfig()
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)
    gv_hist, lam_hist = run_grid(cfg, env)

    print("t,mean_gv,mean_lambda")
    for t in range(len(gv_hist)):
        print(f"{t},{gv_hist[t]:.6f},{lam_hist[t]:.6f}")


if __name__ == "__main__":
    run_demo()
