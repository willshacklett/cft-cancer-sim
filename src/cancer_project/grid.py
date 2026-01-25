"""
Minimal multicell emergence model (systems demo only).

A 2D grid of cells where each site has:
- gv: accumulated strain / risk scalar
- lambda_: local constraint tightness (feedback strength)

Local coupling:
- Healthy cells reinforce neighbor lambda
- Cancer cells erode neighbor lambda
- Rare stochastic events reduce lambda ("mutation" as constraint relaxation)

This is NOT a medical model. It is a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, List, Tuple

import numpy as np

# ---------------------------------------------------------------------
# Import single-cell components (must exist in cancer_project)
# ---------------------------------------------------------------------
try:
    from cancer_project import Environment, HealthyCell, CancerCell
except Exception as e:
    raise ImportError(
        "Could not import Environment / HealthyCell / CancerCell "
        "from cancer_project. Make sure they are exported in __init__.py"
    ) from e


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
@dataclass
class GridConfig:
    n: int = 20              # grid width/height
    steps: int = 60

    # Coupling strengths
    neighbor_strength: float = 0.02
    erosion_strength: float = 0.03

    # Noise / mutation
    mutation_rate: float = 0.002
    mutation_lambda_drop: float = 0.15

    # Bounds
    lambda_min: float = 0.0
    lambda_max: float = 2.0


# ---------------------------------------------------------------------
# Grid Cell Wrapper
# ---------------------------------------------------------------------
class GridCell:
    """
    Lightweight wrapper so every grid site ALWAYS has gv + lambda_.
    """
    def __init__(self, base_cell):
        self.base = base_cell
        self.gv: float = 0.0
        self.lambda_: float = 1.0

    @property
    def is_healthy(self) -> bool:
        return isinstance(self.base, HealthyCell)

    @property
    def is_cancer(self) -> bool:
        return isinstance(self.base, CancerCell)


# ---------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------
class Grid:
    def __init__(self, cfg: GridConfig, env: Environment):
        self.cfg = cfg
        self.env = env

        self.w = cfg.n
        self.h = cfg.n

        # initialize grid
        self.grid: List[List[GridCell]] = [
            [self._init_cell(x, y) for y in range(self.h)]
            for x in range(self.w)
        ]

    # -----------------------------
    # Initialization helpers
    # -----------------------------
    def _init_cell(self, x: int, y: int) -> GridCell:
        # Simple seed: center cancer, rest healthy
        cx, cy = self.w // 2, self.h // 2
        if abs(x - cx) + abs(y - cy) <= 1:
            return GridCell(CancerCell(self.env))
        return GridCell(HealthyCell(self.env))

    # -----------------------------
    # Access helpers
    # -----------------------------
    def cell(self, x: int, y: int) -> GridCell:
        return self.grid[x][y]

    def neighbors(self, x: int, y: int) -> List[GridCell]:
        out = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h:
                out.append(self.grid[nx][ny])
        return out

    def cells_flat(self):
        for row in self.grid:
            for c in row:
                yield c

    # -----------------------------
    # Dynamics
    # -----------------------------
    def step(self):
        cfg = self.cfg

        # 1. Local constraint coupling
        for x in range(self.w):
            for y in range(self.h):
                c = self.cell(x, y)
                for n in self.neighbors(x, y):
                    if c.is_healthy:
                        n.lambda_ += cfg.neighbor_strength
                    elif c.is_cancer:
                        n.lambda_ -= cfg.erosion_strength

        # 2. Mutation / stochastic constraint failure
        for c in self.cells_flat():
            if random.random() < cfg.mutation_rate:
                c.lambda_ -= cfg.mutation_lambda_drop

        # 3. Clamp lambda
        for c in self.cells_flat():
            c.lambda_ = max(cfg.lambda_min, min(cfg.lambda_max, c.lambda_))

        # 4. GV accumulation (strain rises when constraints are weak)
        for c in self.cells_flat():
            strain = max(0.0, 1.0 - c.lambda_)
            c.gv += strain

    # -----------------------------
    # Metrics
    # -----------------------------
    def mean_gv(self) -> float:
        return float(np.mean([c.gv for c in self.cells_flat()]))

    def mean_lambda(self) -> float:
        return float(np.mean([c.lambda_ for c in self.cells_flat()]))

    def gv_field(self) -> np.ndarray:
        return np.array([[self.grid[x][y].gv for y in range(self.h)] for x in range(self.w)])

    def lambda_field(self) -> np.ndarray:
        return np.array([[self.grid[x][y].lambda_ for y in range(self.h)] for x in range(self.w)])


# ---------------------------------------------------------------------
# Demo / Entrypoint
# ---------------------------------------------------------------------
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


# Backwards-compatible alias
run_grid = run_demo


if __name__ == "__main__":
    run_demo()
