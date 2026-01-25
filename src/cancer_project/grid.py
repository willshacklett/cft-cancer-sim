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
from typing import Optional, List, Tuple

import numpy as np
import random

# Optional intervention API (safe if you don't use it)
try:
    from cancer_project.intervention import Intervention, InterventionContext  # type: ignore
except Exception:  # pragma: no cover
    Intervention = None  # type: ignore
    InterventionContext = None  # type: ignore


@dataclass
class GridConfig:
    # grid size (n x n)
    n: int = 20

    # simulation steps
    steps: int = 60

    # initial probability a site is "cancer"
    init_cancer_prob: float = 0.03

    # RNG seed for reproducibility
    seed: int = 7

    # --- coupling dynamics ---
    neighbor_strength: float = 0.02  # healthy reinforcement amount
    cancer_erosion: float = 0.03     # cancer erosion amount

    # --- mutation / constraint drop ---
    mutation_prob: float = 0.01
    mutation_drop: float = 0.15

    # --- bounds ---
    lam_min: float = 0.05
    lam_max: float = 2.0

    # --- GV dynamics knobs (systems-y, not biology) ---
    base_stress: float = 0.015
    lam_protection: float = 0.8   # higher -> strong lam suppresses gv growth more
    gv_diffuse: float = 0.10      # neighbor averaging in gv update (0..1)


@dataclass
class Environment:
    """Simple environment inputs for stress. Keep minimal."""
    toxins: float = 0.2
    oxygen: float = 0.5
    nutrients: float = 0.7


@dataclass
class Site:
    """One grid site."""
    gv: float = 0.0
    lam: float = 2.0
    cancer: bool = False


class Grid:
    def __init__(
        self,
        cfg: GridConfig,
        env: Environment,
        intervention: Optional[object] = None,
    ) -> None:
        self.cfg = cfg
        self.env = env
        self.n = cfg.n
        self.t = 0

        self._rng = random.Random(cfg.seed)

        # initialize cells
        self.cells: List[List[Site]] = []
        for i in range(self.n):
            row: List[Site] = []
            for j in range(self.n):
                is_cancer = self._rng.random() < cfg.init_cancer_prob
                row.append(
                    Site(
                        gv=0.0,
                        lam=cfg.lam_max,  # start tight
                        cancer=is_cancer,
                    )
                )
            self.cells.append(row)

        # intervention (expects .apply(grid, t) OR .apply(grid, ctx))
        self.intervention = intervention

    # ----------------------------
    # Fields / summaries
    # ----------------------------
    def gv_field(self) -> np.ndarray:
        return np.array([[self.cells[i][j].gv for j in range(self.n)] for i in range(self.n)], dtype=float)

    def lam_field(self) -> np.ndarray:
        return np.array([[self.cells[i][j].lam for j in range(self.n)] for i in range(self.n)], dtype=float)

    # Back-compat names some scripts might call
    def lambda_field(self) -> np.ndarray:
        return self.lam_field()

    @property
    def mean_gv(self) -> float:
        g = self.gv_field()
        return float(np.mean(g))

    @property
    def mean_lambda(self) -> float:
        l = self.lam_field()
        return float(np.mean(l))

    # ----------------------------
    # Neighborhood helpers
    # ----------------------------
    def _neighbors(self, i: int, j: int) -> List[Tuple[int, int]]:
        # 4-neighborhood (von Neumann)
        out = []
        if i > 0:
            out.append((i - 1, j))
        if i < self.n - 1:
            out.append((i + 1, j))
        if j > 0:
            out.append((i, j - 1))
        if j < self.n - 1:
            out.append((i, j + 1))
        return out

    # ----------------------------
    # Dynamics
    # ----------------------------
    def step(self) -> None:
        """
        One timestep:
        1) apply local coupling to lam (healthy reinforces, cancer erodes)
        2) apply random mutation drops to lam
        3) update gv given environment + lam protection + neighbor diffusion
        4) apply intervention (if present)
        """
        cfg = self.cfg

        # --- 1) lam coupling update (accumulate deltas then apply) ---
        dlam = np.zeros((self.n, self.n), dtype=float)

        for i in range(self.n):
            for j in range(self.n):
                site = self.cells[i][j]
                neigh = self._neighbors(i, j)
                if not neigh:
                    continue

                if site.cancer:
                    # cancer erodes neighbor constraints
                    for (ni, nj) in neigh:
                        dlam[ni, nj] -= cfg.cancer_erosion
                else:
                    # healthy reinforces neighbor constraints
                    for (ni, nj) in neigh:
                        dlam[ni, nj] += cfg.neighbor_strength

        for i in range(self.n):
            for j in range(self.n):
                s = self.cells[i][j]
                s.lam = float(np.clip(s.lam + dlam[i, j], cfg.lam_min, cfg.lam_max))

        # --- 2) mutation drops ---
        for i in range(self.n):
            for j in range(self.n):
                if self._rng.random() < cfg.mutation_prob:
                    s = self.cells[i][j]
                    s.lam = float(np.clip(s.lam - cfg.mutation_drop, cfg.lam_min, cfg.lam_max))

        # --- 3) gv update ---
        gv_old = self.gv_field()

        # environment "stress" (simple, stable, non-biological)
        # higher toxins, lower oxygen/nutrients -> more stress
        stress = (
            cfg.base_stress
            * (1.0 + self.env.toxins)
            * (1.0 + (1.0 - self.env.oxygen))
            * (1.0 + (1.0 - self.env.nutrients))
        )

        lam = self.lam_field()

        # lam protection: strong lam suppresses gv growth
        protection = 1.0 / (1.0 + cfg.lam_protection * lam)

        gv_growth = stress * protection

        # neighbor diffusion/averaging of gv
        gv_new = gv_old.copy()
        for i in range(self.n):
            for j in range(self.n):
                neigh = self._neighbors(i, j)
                if neigh:
                    neigh_mean = float(np.mean([gv_old[ni, nj] for (ni, nj) in neigh]))
                else:
                    neigh_mean = float(gv_old[i, j])

                mixed = (1.0 - cfg.gv_diffuse) * gv_old[i, j] + cfg.gv_diffuse * neigh_mean
                gv_new[i, j] = max(0.0, mixed + gv_growth[i, j])

        # commit gv_new back to sites
        for i in range(self.n):
            for j in range(self.n):
                self.cells[i][j].gv = float(gv_new[i, j])

        # --- 4) intervention hook ---
        if self.intervention is not None:
            # Support either style:
            #  - intervention.apply(grid, t)
            #  - intervention.apply(grid, InterventionContext(t=...))
            try:
                self.intervention.apply(self, self.t)  # type: ignore[attr-defined]
            except TypeError:
                if InterventionContext is not None:
                    ctx = InterventionContext(t=self.t)  # type: ignore[call-arg]
                    self.intervention.apply(self, ctx)  # type: ignore[attr-defined]
                else:
                    raise

        self.t += 1


# ----------------------------
# CLI demo / script back-compat
# ----------------------------
def run_demo() -> None:
    cfg = GridConfig()
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)
    grid = Grid(cfg, env)

    print("t,mean_gv,mean_lambda")
    for _ in range(cfg.steps):
        grid.step()
        print(f"{grid.t},{grid.mean_gv:.6f},{grid.mean_lambda:.6f}")


# Backwards-compatible alias for scripts expecting run_grid()
run_grid = run_demo


if __name__ == "__main__":
    run_demo()
