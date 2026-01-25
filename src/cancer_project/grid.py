from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random

# We reuse your existing single-cell logic and GV scoring.
from cancer_project import Environment, HealthyCell, CancerCell, gv_score


@dataclass
class GridConfig:
    n: int = 10                  # grid size n x n
    steps: int = 60              # timesteps
    seed: int = 1337

    # coupling strength: how much neighbors affect a cell's constraint tightness (lambda)
    epsilon: float = 0.08

    # how strongly lambda reduces GV each step (feedback tightness)
    constraint_strength: float = 1.0

    # mutation: rare lambda drop events (turning constraints weaker)
    mutation_rate: float = 0.003
    mutation_drop: float = 0.15

    # initial cancer seeding
    cancer_fraction: float = 0.05

    # clamp lambda bounds
    lambda_min: float = 0.05
    lambda_max: float = 1.50


class Grid:
    """
    Minimal multicell emergence model:
    - Each site holds a cell (HealthyCell or CancerCell) + a constraint tightness λ (lambda).
    - Cells accumulate strain via existing single-cell step() logic.
    - GV update per cell:
        GV_{t+1} = GV_t + strain_sources - (lambda * constraint_strength * GV_t)
      where "strain_sources" uses your gv_score components (atp, damage, arrest_steps, divisions).
    - Local coupling:
        Healthy cells reinforce neighbor λ (move toward neighbor mean).
        Cancer cells erode neighbor λ (push away from neighbor mean).
    """

    def __init__(self, cfg: GridConfig, env: Environment):
        self.cfg = cfg
        self.env = env

        random.seed(cfg.seed)

        self.cells: List[List[object]] = []
        self.lmbd: List[List[float]] = []
        self.gv: List[List[float]] = []

        for _y in range(cfg.n):
            row_cells = []
            row_l = []
            row_gv = []
            for _x in range(cfg.n):
                if random.random() < cfg.cancer_fraction:
                    row_cells.append(CancerCell())
                    row_l.append(0.45)   # cancer starts with looser constraints
                else:
                    row_cells.append(HealthyCell())
                    row_l.append(1.10)   # healthy starts with tighter constraints
                row_gv.append(0.0)
            self.cells.append(row_cells)
            self.lmbd.append(row_l)
            self.gv.append(row_gv)

    def _neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        n = self.cfg.n
        coords = []
        if x > 0: coords.append((x - 1, y))
        if x < n - 1: coords.append((x + 1, y))
        if y > 0: coords.append((x, y - 1))
        if y < n - 1: coords.append((x, y + 1))
        return coords

    def _clamp_lambda(self, v: float) -> float:
        return max(self.cfg.lambda_min, min(self.cfg.lambda_max, v))

    def step(self) -> None:
        """
        One timestep:
        1) Step each cell (updates atp/damage/arrest/divisions via existing code)
        2) Update GV using a simple strain - feedback rule
        3) Apply local coupling on lambda (healthy reinforces, cancer erodes)
        4) Apply rare mutation events that reduce lambda
        """
        n = self.cfg.n

        # 1) step cells
        for y in range(n):
            for x in range(n):
                cell = self.cells[y][x]
                cell.step(self.env)

        # 2) update GV from current cell state (strain sources) + damping from lambda
        for y in range(n):
            for x in range(n):
                cell = self.cells[y][x]
                g_prev = self.gv[y][x]

                # "strain_sources" proxy using your gv_score (keeps it consistent)
                strain = gv_score(cell.atp, cell.damage, cell.arrest_steps, cell.divisions)

                # feedback term: tighter lambda = stronger reduction of existing GV
                lam = self.lmbd[y][x]
                g_next = g_prev + strain - (lam * self.cfg.constraint_strength * g_prev)

                # keep GV non-negative for interpretability
                self.gv[y][x] = max(0.0, g_next)

        # 3) local coupling on lambda (compute into a new grid to avoid order bias)
        new_lmbd = [[self.lmbd[y][x] for x in range(n)] for y in range(n)]

        for y in range(n):
            for x in range(n):
                neigh = self._neighbors(x, y)
                if not neigh:
                    continue

                neigh_mean = sum(self.lmbd[yy][xx] for (xx, yy) in neigh) / len(neigh)
                lam = self.lmbd[y][x]
                delta = neigh_mean - lam

                cell = self.cells[y][x]
                is_cancer = isinstance(cell, CancerCell)

                if is_cancer:
                    # cancer: erode constraints (push away from healthy neighborhood)
                    new_lmbd[y][x] = self._clamp_lambda(lam - self.cfg.epsilon * abs(delta))
                else:
                    # healthy: reinforce constraints (move toward neighborhood coherence)
                    new_lmbd[y][x] = self._clamp_lambda(lam + self.cfg.epsilon * delta)

        self.lmbd = new_lmbd

        # 4) rare mutation events that reduce lambda
        for y in range(n):
            for x in range(n):
                if random.random() < self.cfg.mutation_rate:
                    self.lmbd[y][x] = self._clamp_lambda(self.lmbd[y][x] - self.cfg.mutation_drop)

    def mean_gv(self) -> float:
        n = self.cfg.n
        return sum(self.gv[y][x] for y in range(n) for x in range(n)) / (n * n)

    def mean_lambda(self) -> float:
        n = self.cfg.n
        return sum(self.lmbd[y][x] for y in range(n) for x in range(n)) / (n * n)


def run_demo() -> None:
    """
    Prints a simple emergence signal:
    - mean GV over time
    - mean lambda over time
    """
    cfg = GridConfig()
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)
    grid = Grid(cfg, env)

    print("t,mean_gv,mean_lambda")
    for t in range(cfg.steps):
        grid.step()
        print(f"{t},{grid.mean_gv():.6f},{grid.mean_lambda():.6f}")


if __name__ == "__main__":
    run_demo()
