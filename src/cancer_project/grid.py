"""
Minimal multicell emergence model (GV + constraint tightness lambda).

Run:
  pip install -e .
  python -m cancer_project.grid | head -n 10

This is a systems-level demonstration only. No medical claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import List, Tuple, Dict, Any

# Imports from your single-cell model
from cancer_project import Environment, HealthyCell, CancerCell, gv_score


@dataclass
class GridConfig:
    width: int = 10
    height: int = 10
    steps: int = 60

    # Initial composition
    init_cancer_frac: float = 0.05  # start with a few cancer cells

    # Constraint tightness (lambda)
    lambda_init: float = 1.0
    lambda_min: float = 0.0
    lambda_max: float = 1.0

    # Local coupling strength (neighbors)
    neighbor_strength: float = 0.01  # how strongly cells influence neighbors
    healthy_reinforce: float = 1.0   # healthy pushes lambda up
    cancer_erode: float = 1.5        # cancer pushes lambda down (stronger than reinforce)

    # Stochastic mutation (healthy -> cancer)
    mutation_rate: float = 0.001     # per-cell per-step chance
    mutation_lambda_drop: float = 0.25  # drop lambda when a cell flips to cancer

    # RNG
    seed: int = 7


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _safe_cell_gv(cell) -> float:
    """
    Uses gv_score(cell...) if available in your project;
    otherwise falls back to a simple proxy.
    """
    try:
        atp = getattr(cell, "atp", 1.0)
        damage = getattr(cell, "damage", 0.0)
        arrest_steps = getattr(cell, "arrest_steps", 0)
        divisions = getattr(cell, "divisions", 0)
        return float(gv_score(atp, damage, arrest_steps, divisions))
    except Exception:
        # Fallback: higher damage/arrest/divisions => higher GV proxy
        damage = float(getattr(cell, "damage", 0.0))
        arrest_steps = float(getattr(cell, "arrest_steps", 0.0))
        divisions = float(getattr(cell, "divisions", 0.0))
        return 0.1 * damage + 0.01 * arrest_steps + 0.02 * divisions


def _make_cell(cell_cls, env: Environment):
    """
    Tries a few common constructor patterns without crashing.
    """
    for args in ((), (env,)):
        try:
            return cell_cls(*args)
        except TypeError:
            continue
    # Last resort: just call with no args (will raise if impossible)
    return cell_cls()


class Grid:
    def __init__(self, cfg: GridConfig, env: Environment):
        self.cfg = cfg
        self.env = env

        random.seed(cfg.seed)

        self.cells: List[List[object]] = []
        for y in range(cfg.height):
            row = []
            for x in range(cfg.width):
                if random.random() < cfg.init_cancer_frac:
                    c = _make_cell(CancerCell, env)
                else:
                    c = _make_cell(HealthyCell, env)

                self._ensure_lambda(c)

                # If you want initial “pockets” weaker, uncomment:
                # if isinstance(c, CancerCell):
                #     c.lambda_ = _clamp(c.lambda_ - cfg.mutation_lambda_drop, cfg.lambda_min, cfg.lambda_max)

                row.append(c)
            self.cells.append(row)

    def _ensure_lambda(self, cell) -> None:
        # Guarantee lambda_ exists on every cell
        if not hasattr(cell, "lambda_"):
            cell.lambda_ = float(self.cfg.lambda_init)
        # Clamp it
        cell.lambda_ = _clamp(float(cell.lambda_), self.cfg.lambda_min, self.cfg.lambda_max)

    def neighbors4(self, x: int, y: int) -> List[Tuple[int, int]]:
        out = []
        if x > 0:
            out.append((x - 1, y))
        if x < self.cfg.width - 1:
            out.append((x + 1, y))
        if y > 0:
            out.append((x, y - 1))
        if y < self.cfg.height - 1:
            out.append((x, y + 1))
        return out

    def step(self) -> None:
        """
        One timestep:
          1) each cell takes its own step(env)
          2) rare mutation flips a cell to cancer (with lambda drop)
          3) neighbor coupling adjusts lambda_ (healthy reinforces, cancer erodes)
        """
        # 1) cell-internal dynamics
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                c = self.cells[y][x]
                self._ensure_lambda(c)

                step_fn = getattr(c, "step", None)
                if callable(step_fn):
                    step_fn(self.env)

        # 2) stochastic mutation: Healthy -> Cancer
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                c = self.cells[y][x]
                self._ensure_lambda(c)

                if not isinstance(c, CancerCell) and random.random() < self.cfg.mutation_rate:
                    newc = _make_cell(CancerCell, self.env)
                    self._ensure_lambda(newc)
                    # inherit & weaken constraint tightness
                    newc.lambda_ = _clamp(
                        float(getattr(c, "lambda_", self.cfg.lambda_init)) - self.cfg.mutation_lambda_drop,
                        self.cfg.lambda_min,
                        self.cfg.lambda_max,
                    )
                    self.cells[y][x] = newc

        # 3) neighbor coupling (apply as simultaneous deltas)
        delta: List[List[float]] = [[0.0 for _ in range(self.cfg.width)] for __ in range(self.cfg.height)]

        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                c = self.cells[y][x]
                self._ensure_lambda(c)

                # Influence sign/magnitude depends on type
                if isinstance(c, CancerCell):
                    influence = -self.cfg.neighbor_strength * self.cfg.cancer_erode
                else:
                    influence = +self.cfg.neighbor_strength * self.cfg.healthy_reinforce

                for nx, ny in self.neighbors4(x, y):
                    delta[ny][nx] += influence

        # Apply deltas to neighbor lambda_
        for y in range(self.cfg.height):
            for x in range(self.cfg.width):
                c = self.cells[y][x]
                self._ensure_lambda(c)
                c.lambda_ = _clamp(float(c.lambda_) + float(delta[y][x]), self.cfg.lambda_min, self.cfg.lambda_max)

    @property
    def mean_lambda(self) -> float:
        s = 0.0
        n = 0
        for row in self.cells:
            for c in row:
                self._ensure_lambda(c)
                s += float(c.lambda_)
                n += 1
        return s / max(1, n)

    @property
    def mean_gv(self) -> float:
        s = 0.0
        n = 0
        for row in self.cells:
            for c in row:
                s += _safe_cell_gv(c)
                n += 1
        return s / max(1, n)


def run(cfg: GridConfig | None = None, env: Environment | None = None) -> List[Dict[str, Any]]:
    """
    Returns a history list of dicts:
      [{"t":0,"mean_gv":...,"mean_lambda":...}, ...]
    """
    cfg = cfg or GridConfig()
    env = env or Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)

    grid = Grid(cfg, env)
    hist: List[Dict[str, Any]] = []

    for t in range(cfg.steps):
        grid.step()
        hist.append({"t": t, "mean_gv": grid.mean_gv, "mean_lambda": grid.mean_lambda})

    return hist


def run_demo() -> None:
    """
    Prints a simple emergence signal:
      t,mean_gv,mean_lambda
    """
    cfg = GridConfig()
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)
    grid = Grid(cfg, env)

    print("t,mean_gv,mean_lambda")
    for t in range(cfg.steps):
        grid.step()
        print(f"{t},{grid.mean_gv:.6f},{grid.mean_lambda:.6f}")


# Backwards-compatible alias for scripts expecting run_grid()
run_grid = run


if __name__ == "__main__":
    run_demo()
