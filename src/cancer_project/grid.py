"""
Minimal multicell emergence model.
Systems-level demo only.
"""

from dataclasses import dataclass
import random
import numpy as np

from cancer_project.env import Environment
from cancer_project.healthy_cell import HealthyCell
from cancer_project.cancer_cell import CancerCell


@dataclass
class GridConfig:
    width: int = 10
    height: int = 10
    steps: int = 60
    mutation_rate: float = 0.001
    neighbor_strength: float = 0.05


class Grid:
    def __init__(self, cfg: GridConfig, env: Environment):
        self.cfg = cfg
        self.env = env
        self.cells = []

        for y in range(cfg.height):
            row = []
            for x in range(cfg.width):
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
        for row in self.cells:
            for cell in row:
                cell.step(self.env)

        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                cell = self.cells[y][x]
                for n in self.neighbors(x, y):
                    if isinstance(cell, HealthyCell):
                        n.lambda_ += self.cfg.neighbor_strength
                    else:
                        n.lambda_ -= self.cfg.neighbor_strength

        for row in self.cells:
            for cell in row:
                if isinstance(cell, HealthyCell):
                    if random.random() < self.cfg.mutation_rate:
                        cell.__class__ = CancerCell

    def mean_gv(self):
        return float(np.mean([c.gv for r in self.cells for c in r]))

    def mean_lambda(self):
        return float(np.mean([c.lambda_ for r in self.cells for c in r]))


def run_demo():
    cfg = GridConfig()
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)
    grid = Grid(cfg, env)

    print("t,mean_gv,mean_lambda")
    for t in range(cfg.steps):
        grid.step()
        print(f"{t},{grid.mean_gv():.6f},{grid.mean_lambda():.6f}")


# 🔑 THIS IS THE IMPORTANT LINE
run_grid = run_demo


if __name__ == "__main__":
    run_demo()
