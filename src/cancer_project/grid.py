"""
Minimal multicell emergence model.

Purpose:
- Demonstrate emergence from constraint coupling alone
- No biological or medical claims
- Systems-level toy model
"""

from dataclasses import dataclass
import random
import numpy as np

from cancer_project.env import Environment
from cancer_project.healthy_cell import HealthyCell
from cancer_project.cancer_cell import CancerCell


# ----------------------------
# Configuration
# ----------------------------

@dataclass
class GridConfig:
    width: int = 10
    height: int = 10
    steps: int = 60
    mutation_rate: float = 0.001  # chance per cell per step
    neighbor_strength: float = 0.05


# ----------------------------
# Grid
# ----------------------------

class Grid:
    def __init__(self, cfg: GridConfig, env: Environment):
        self.cfg = cfg
        self.env = env

        self.cells = []
        for y in range(cfg.height):
            row = []
            for x in range(cfg.width):
                # Start mostly healthy, seed a few cancer cells
                if random.random() < 0.05:
                    row.append(CancerCell())
                else:
                    row.append(HealthyCell())
            self.cells.append(row)

    def neighbors(self, x, y):
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.cfg.width and 0 <= ny < self.cfg.height:
                yield self.cells[ny][nx]

    def step(self):
        # Step all cells
        for row in self.cells:
            for cell in row:
                cell.step(self.env)

        # Constraint coupling between neighbors
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                cell = self.cells[y][x]
                for n in self.neighbors(x, y):
                    if isinstance(cell, HealthyCell):
                        n.lambda_ += self.cfg.neighbor_strength
                    else:
                        n.lambda_ -= self.cfg.neighbor_strength

        # Rare stochastic mutation: healthy → cancer
        for row in self.cells:
            for cell in row:
                if isinstance(cell, HealthyCell):
                    if random.random() < self.cfg.mutation_rate:
                        cell.__class__ = CancerCell

    def mean_gv(self):
        vals = [
            cell.gv for row in self.cells for cell in row
            if hasattr(cell, "gv")
        ]
        return float(np.mean(vals)) if vals else 0.0

    def mean_lambda(self):
        vals = [
            cell.lambda_ for row in self.cells for cell in row
            if hasattr(cell, "lambda_")
        ]
        return float(np.mean(vals)) if vals else 0.0


# ----------------------------
# Demo / CLI entrypoint
# ----------------------------

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


# 🔑 Backwards-compatible alias
run_grid = run_demo


if __name__ == "__main__":
    run_demo()
