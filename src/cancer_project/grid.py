"""
Minimal multicell emergence model (systems demo only).

A 2D grid of sites. Each site has:
- a Cell object (HealthyCell or CancerCell) from the single-cell model
- gv: derived scalar "strain / risk" (computed via gv_score from cell state)
- lam: local constraint tightness (feedback strength) tracked on the grid

Local coupling:
- Healthy sites reinforce neighbor lam
- Cancer sites erode neighbor lam
- Rare stochastic "mutation" events reduce lam

This is NOT a medical model. It is a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, Tuple, Any, Dict

import numpy as np

# Single-cell components (must exist in cancer_project)
from cancer_project import Environment, HealthyCell, CancerCell
from cancer_project.gv import gv_score


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class GridConfig:
    n: int = 20
    steps: int = 60

    # initialization
    init_cancer_prob: float = 0.03
    seed: int = 7

    # lambda / constraint field bounds
    lam_init: float = 1.0
    lam_min: float = 0.1
    lam_max: float = 2.0

    # coupling dynamics
    neighbor_strength: float = 0.02   # per-step neighbor influence magnitude
    mutation_prob: float = 0.002      # per-site per-step chance of lam drop
    mutation_drop: float = 0.15       # amount lam drops on mutation

    # how strongly local lam feeds back into environment for that site
    # (kept simple and bounded so it doesn't explode)
    lam_env_gain: float = 0.15        # higher lam => slightly better effective environment


# -----------------------------
# Helpers
# -----------------------------

def _is_cancer(cell: Any) -> bool:
    return cell.__class__.__name__.lower().startswith("cancer")


def _cell_snapshot(cell: Any) -> Dict[str, Any]:
    """
    Robustly extract state needed for gv_score without assuming too much.
    """
    # Some cells expose snapshot(); if present, use it.
    if hasattr(cell, "snapshot") and callable(cell.snapshot):
        try:
            snap = cell.snapshot()
            if isinstance(snap, dict):
                return snap
        except Exception:
            pass

    # Fallback to common attributes
    return {
        "atp": getattr(cell, "atp", 0.0),
        "damage": getattr(cell, "damage", 0.0),
        "arrest_steps": getattr(cell, "arrest_steps", 0),
        "divisions": getattr(cell, "divisions", 0),
    }


def _make_cell(kind: str) -> Any:
    """
    Instantiate HealthyCell/CancerCell in a way that survives minor signature changes.
    """
    cls = HealthyCell if kind == "healthy" else CancerCell

    # Try a few common constructor patterns.
    for args, kwargs in [
        ((), {}),  # default constructor
        ((), {"rng": random.Random()}),
        ((random.Random(),), {}),
    ]:
        try:
            return cls(*args, **kwargs)
        except TypeError:
            continue

    # If none worked, last attempt with no args (will raise a useful error)
    return cls()


# -----------------------------
# Grid model
# -----------------------------

class Grid:
    def __init__(self, cfg: GridConfig, env: Optional[Environment] = None):
        self.cfg = cfg
        self.n = cfg.n
        self.rng = random.Random(cfg.seed)

        self.env = env if env is not None else Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)

        # Cells
        self.cells = np.empty((self.n, self.n), dtype=object)
        for i in range(self.n):
            for j in range(self.n):
                if self.rng.random() < cfg.init_cancer_prob:
                    self.cells[i, j] = _make_cell("cancer")
                else:
                    self.cells[i, j] = _make_cell("healthy")

        # Lambda field
        self.lam = np.full((self.n, self.n), float(cfg.lam_init), dtype=float)

        self.t = 0

    def neighbors4(self, i: int, j: int):
        if i > 0:
            yield (i - 1, j)
        if i < self.n - 1:
            yield (i + 1, j)
        if j > 0:
            yield (i, j - 1)
        if j < self.n - 1:
            yield (i, j + 1)

    def _effective_env(self, lam_ij: float) -> Environment:
        """
        Very simple "feedback recoverability" hook:
        higher lam gives a slightly better effective environment locally,
        without hardcoding biology.
        """
        g = self.cfg.lam_env_gain
        # Keep it bounded and gentle
        boost = 1.0 + g * (lam_ij - 1.0)
        boost = max(0.5, min(1.5, boost))

        return Environment(
            toxins=max(0.0, min(1.0, self.env.toxins / boost)),
            oxygen=max(0.0, min(1.0, self.env.oxygen * boost)),
            nutrients=max(0.0, min(1.0, self.env.nutrients * boost)),
        )

    def step(self) -> None:
        """
        One grid timestep:
        1) Step each cell with its local effective environment (keyword args to avoid mis-binding).
        2) Update lam via local coupling (healthy reinforces, cancer erodes) + rare mutation drops.
        """
        cfg = self.cfg
        n = self.n

        # 1) Step cells
        for i in range(n):
            for j in range(n):
                cell = self.cells[i, j]
                local_env = self._effective_env(self.lam[i, j])

                if hasattr(cell, "step") and callable(cell.step):
                    # Use keywords so we never accidentally pass env as dt, etc.
                    try:
                        cell.step(env=local_env, dt=1.0, rng=self.rng)
                    except TypeError:
                        # Some versions might not accept dt/rng
                        try:
                            cell.step(env=local_env)
                        except TypeError:
                            # Worst-case: older signature step(env, dt)
                            cell.step(local_env, 1.0)

        # 2) Update lambda field (compute deltas then apply)
        dlam = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                cell = self.cells[i, j]
                s = cfg.neighbor_strength

                if _is_cancer(cell):
                    # cancer erodes neighbors
                    influence = -s
                else:
                    # healthy reinforces neighbors
                    influence = +s

                for (ni, nj) in self.neighbors4(i, j):
                    dlam[ni, nj] += influence

                # rare mutation: local lam drop
                if self.rng.random() < cfg.mutation_prob:
                    dlam[i, j] -= cfg.mutation_drop

        self.lam += dlam
        np.clip(self.lam, cfg.lam_min, cfg.lam_max, out=self.lam)

        self.t += 1

    # ---- fields / metrics ----

    def gv_field(self) -> np.ndarray:
        """
        Compute GV over the grid from cell state.
        """
        gv = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                snap = _cell_snapshot(self.cells[i, j])
                gv[i, j] = float(
                    gv_score(
                        atp=float(snap.get("atp", 0.0)),
                        damage=float(snap.get("damage", 0.0)),
                        arrest_steps=int(snap.get("arrest_steps", 0)),
                        divisions=int(snap.get("divisions", 0)),
                    )
                )
        return gv

    def lam_field(self) -> np.ndarray:
        return self.lam.copy()

    @property
    def mean_gv(self) -> float:
        g = self.gv_field()
        return float(np.mean(g))

    @property
    def mean_lambda(self) -> float:
        return float(np.mean(self.lam))


# -----------------------------
# Entrypoints
# -----------------------------

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
        print(f"{t},{grid.mean_gv:.6f},{grid.mean_lambda:.6f}")


def run() -> None:
    run_demo()


# Backwards-compatible alias for scripts expecting run_grid()
run_grid = run_demo


if __name__ == "__main__":
    run_demo()
