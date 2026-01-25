"""
Minimal multicell emergence model (systems demo only).

A 2D grid of sites. Each site has:
- a Cell object (HealthyCell or CancerCell)
- gv: a derived "strain/risk" scalar (computed from cell state via gv_score)
- lam: local constraint tightness / feedback strength (scalar field)

Local coupling:
- Healthy sites reinforce neighbor lam
- Cancer sites erode neighbor lam
- Rare stochastic events reduce lam ("mutation" as constraint relaxation)

This is NOT a medical model. It is a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, Tuple

import numpy as np

# Import single-cell components (must exist in cancer_project)
try:
    from cancer_project import Environment, HealthyCell, CancerCell
except Exception as e:
    raise ImportError(
        "Could not import Environment / HealthyCell / CancerCell from cancer_project. "
        "Make sure they exist and are exported in cancer_project/__init__.py"
    ) from e

# GV scoring function (must exist in cancer_project/gv.py)
try:
    from cancer_project.gv import gv_score
except Exception as e:
    raise ImportError(
        "Could not import gv_score from cancer_project.gv. "
        "Make sure cancer_project/gv.py exports gv_score."
    ) from e


@dataclass
class GridConfig:
    # grid shape
    n: int = 20
    steps: int = 60

    # initialization
    init_cancer_prob: float = 0.03
    seed: Optional[int] = 7

    # constraint field (lam) parameters
    lam_init: float = 1.0
    lam_min: float = 0.0
    lam_max: float = 2.0

    # coupling strength to neighbors per step
    neighbor_strength: float = 0.03

    # probability per site per step of a "mutation" event that reduces lam
    mutation_prob: float = 0.002
    mutation_drop: float = 0.10

    # optional small diffusion/smoothing of lam each step (keeps fields nicer)
    lam_smoothing: float = 0.10  # 0 disables, typical 0.05–0.2


class Grid:
    def __init__(self, cfg: GridConfig, env: Optional[Environment] = None):
        self.cfg = cfg
        self.n = int(cfg.n)

        # reproducible randomness
        self.rng = random.Random(cfg.seed)

        # environment used by cell.step(...)
        self.env = env if env is not None else Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)

        # cell grid
        self.cells: np.ndarray = np.empty((self.n, self.n), dtype=object)
        for i in range(self.n):
            for j in range(self.n):
                if self.rng.random() < cfg.init_cancer_prob:
                    self.cells[i, j] = CancerCell()
                else:
                    self.cells[i, j] = HealthyCell()

        # fields
        self.gv: np.ndarray = np.zeros((self.n, self.n), dtype=float)
        self.lam: np.ndarray = np.full((self.n, self.n), float(cfg.lam_init), dtype=float)

        # time
        self.t: int = 0

        # initialize gv once from initial cell states
        self._recompute_gv()

    # ---------------------------
    # neighborhood helpers
    # ---------------------------
    def _neighbors4(self, i: int, j: int):
        if i > 0:
            yield (i - 1, j)
        if i < self.n - 1:
            yield (i + 1, j)
        if j > 0:
            yield (i, j - 1)
        if j < self.n - 1:
            yield (i, j + 1)

    # ---------------------------
    # core dynamics
    # ---------------------------
    def step(self) -> None:
        """
        One simulation step:
        1) advance all cells using env
        2) recompute gv field from cell state
        3) update lam field via local coupling + mutation + optional smoothing
        """
        # 1) cell updates
        for i in range(self.n):
            for j in range(self.n):
                # IMPORTANT: HealthyCell.step requires env (your error earlier)
                self.cells[i, j].step(self.env, dt=1.0, rng=self.rng)

        # 2) compute gv from state
        self._recompute_gv()

        # 3) update lam
        self._update_lam()

        self.t += 1

    def _recompute_gv(self) -> None:
        """
        Compute gv[i,j] from each cell snapshot using gv_score(atp, damage, arrest_steps, divisions).
        We do NOT assume cells store gv.
        """
        for i in range(self.n):
            for j in range(self.n):
                cell = self.cells[i, j]
                snap = cell.snapshot() if hasattr(cell, "snapshot") else {}

                atp = float(snap.get("atp", getattr(cell, "atp", 0.0)))
                damage = float(snap.get("damage", getattr(cell, "damage", 0.0)))
                arrest_steps = int(snap.get("arrest_steps", getattr(cell, "arrest_steps", 0)))
                divisions = int(snap.get("divisions", getattr(cell, "divisions", 0)))

                self.gv[i, j] = float(gv_score(atp, damage, arrest_steps, divisions))

    def _update_lam(self) -> None:
        """
        Local coupling:
        - Healthy cells reinforce neighbor lam
        - Cancer cells erode neighbor lam
        - Mutation randomly drops lam at sites
        - Optional smoothing (diffusion-like)
        """
        cfg = self.cfg
        lam_next = self.lam.copy()

        # neighbor coupling
        for i in range(self.n):
            for j in range(self.n):
                cell = self.cells[i, j]
                is_cancer = isinstance(cell, CancerCell)

                delta = -cfg.neighbor_strength if is_cancer else cfg.neighbor_strength

                for ni, nj in self._neighbors4(i, j):
                    lam_next[ni, nj] += delta

        # mutation events: random local drops
        if cfg.mutation_prob > 0.0 and cfg.mutation_drop > 0.0:
            for i in range(self.n):
                for j in range(self.n):
                    if self.rng.random() < cfg.mutation_prob:
                        lam_next[i, j] -= cfg.mutation_drop

        # optional smoothing (keeps field more "tissue-like")
        if cfg.lam_smoothing and cfg.lam_smoothing > 0.0:
            a = float(cfg.lam_smoothing)
            sm = lam_next.copy()
            for i in range(self.n):
                for j in range(self.n):
                    neigh = [(i, j)]
                    neigh.extend(list(self._neighbors4(i, j)))
                    avg = sum(lam_next[x, y] for x, y in neigh) / len(neigh)
                    sm[i, j] = (1.0 - a) * lam_next[i, j] + a * avg
            lam_next = sm

        # clamp
        lam_next = np.clip(lam_next, cfg.lam_min, cfg.lam_max)

        self.lam = lam_next

    # ---------------------------
    # observables / helpers
    # ---------------------------
    def gv_field(self) -> np.ndarray:
        return self.gv.copy()

    def lam_field(self) -> np.ndarray:
        return self.lam.copy()

    # Back-compat for scripts that call lambda_field()
    def lambda_field(self) -> np.ndarray:
        return self.lam_field()

    def mean_gv(self) -> float:
        return float(np.mean(self.gv))

    def mean_lambda(self) -> float:
        return float(np.mean(self.lam))

    # ---------------------------
    # convenience runners
    # ---------------------------
    def run(self, steps: Optional[int] = None) -> "Grid":
        """
        Run for steps (default cfg.steps) and return self.
        """
        total = int(self.cfg.steps if steps is None else steps)
        for _ in range(total):
            self.step()
        return self


def run_demo() -> None:
    """
    Prints a simple emergence signal:
    t, mean_gv, mean_lambda
    """
    cfg = GridConfig()
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)
    grid = Grid(cfg, env=env)

    print("t,mean_gv,mean_lambda")
    for t in range(cfg.steps + 1):
        if t > 0:
            grid.step()
        print(f"{t},{grid.mean_gv():.6f},{grid.mean_lambda():.6f}")


# Backwards-compatible alias for scripts expecting run_grid()
run_grid = run_demo


if __name__ == "__main__":
    run_demo()
