"""
Minimal multicell emergence model (systems demo only).

A 2D grid of sites. Each site has:
- a Cell (HealthyCell or CancerCell)
- lam: local constraint tightness (feedback strength)

Local coupling:
- Healthy sites reinforce neighbor lam
- Cancer sites erode neighbor lam
- Rare stochastic "mutation" events reduce lam

This is NOT a medical model. It is a systems-level demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional, Tuple, List, Any, Dict

import numpy as np

# Import single-cell components (must exist in cancer_project)
try:
    from cancer_project import Environment, HealthyCell, CancerCell
except Exception as e:
    raise ImportError(
        "Could not import Environment / HealthyCell / CancerCell from cancer_project. "
        "Make sure they exist and are exported in cancer_project/__init__.py"
    ) from e

# Optional GV scoring helper (preferred)
try:
    from cancer_project.gv import gv_score as _gv_score
except Exception:
    _gv_score = None


@dataclass
class GridConfig:
    # geometry / time
    n: int = 20
    steps: int = 60
    seed: int = 7

    # initialization
    init_cancer_prob: float = 0.03

    # constraint field (lam)
    lam_init: float = 2.0
    lam_min: float = 0.0
    lam_max: float = 2.0

    # coupling strengths
    neighbor_strength: float = 0.02     # healthy reinforces neighbors
    cancer_erosion: float = 0.03        # cancer erodes neighbors

    # stochastic degradation ("mutation")
    mutation_prob: float = 0.002
    mutation_erosion: float = 0.05

    # integration
    dt: float = 1.0


class Grid:
    """
    Grid owns:
    - self.cells[i][j] -> HealthyCell or CancerCell
    - self.lam[i][j]   -> local constraint tightness
    - self.env         -> shared Environment passed into each cell.step(...)
    """

    def __init__(self, cfg: GridConfig, env: Optional[Environment] = None):
        self.cfg = cfg
        self.n = cfg.n
        self.rng = random.Random(cfg.seed)

        self.env = env if env is not None else Environment()

        # cell lattice
        self.cells: List[List[Any]] = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if self.rng.random() < cfg.init_cancer_prob:
                    row.append(CancerCell())
                else:
                    row.append(HealthyCell())
            self.cells.append(row)

        # constraint field
        self.lam = np.full((self.n, self.n), float(cfg.lam_init), dtype=float)

        # optional interventions (you can attach later)
        self.interventions: List[Any] = []

    # ----------------------------
    # Neighborhood helpers
    # ----------------------------
    def neighbors4(self, i: int, j: int) -> List[Tuple[int, int]]:
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
    # Fields for plotting
    # ----------------------------
    def lam_field(self) -> np.ndarray:
        return self.lam.copy()

    def gv_field(self) -> np.ndarray:
        """
        Compute GV per cell from each cell's snapshot.
        We DO NOT rely on cell.gv existing (because it doesn’t in your code).
        """
        gv = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                snap = self._safe_snapshot(self.cells[i][j])
                gv[i, j] = float(self._score_from_snapshot(snap))
        return gv

    def _safe_snapshot(self, cell: Any) -> Dict[str, Any]:
        if hasattr(cell, "snapshot") and callable(cell.snapshot):
            return cell.snapshot()
        # fallback if snapshot not present
        return {
            "atp": getattr(cell, "atp", 0.0),
            "damage": getattr(cell, "damage", 0.0),
            "arrest_steps": getattr(cell, "arrest_steps", 0),
            "divisions": getattr(cell, "divisions", 0),
        }

    def _score_from_snapshot(self, snap: Dict[str, Any]) -> float:
        atp = float(snap.get("atp", 0.0))
        damage = float(snap.get("damage", 0.0))
        arrest = int(snap.get("arrest_steps", 0))
        divs = int(snap.get("divisions", 0))

        if _gv_score is not None:
            return float(_gv_score(atp=atp, damage=damage, arrest_steps=arrest, divisions=divs))

        # simple fallback if gv_score isn't available
        # (higher damage + more arrests/divisions -> higher "risk")
        return max(0.0, damage) + 0.01 * arrest + 0.02 * divs - 0.001 * atp

    # ----------------------------
    # Main dynamics
    # ----------------------------
    def step(self, t: int = 0) -> None:
        # 1) advance cell internal dynamics (IMPORTANT: pass env correctly)
        dt = float(self.cfg.dt)
        for i in range(self.n):
            for j in range(self.n):
                cell = self.cells[i][j]
                # HealthyCell.step(env, dt=..., rng=...) is what your files define
                if hasattr(cell, "step"):
                    cell.step(self.env, dt=dt, rng=self.rng)

        # 2) update lam by local coupling
        dlam = np.zeros((self.n, self.n), dtype=float)

        for i in range(self.n):
            for j in range(self.n):
                cell = self.cells[i][j]

                is_cancer = isinstance(cell, CancerCell)
                is_healthy = isinstance(cell, HealthyCell)

                if is_healthy:
                    amt = float(self.cfg.neighbor_strength)
                elif is_cancer:
                    amt = -float(self.cfg.cancer_erosion)
                else:
                    amt = 0.0

                if amt != 0.0:
                    for (ni, nj) in self.neighbors4(i, j):
                        dlam[ni, nj] += amt

        # 3) stochastic degradation events (mutation-like)
        for i in range(self.n):
            for j in range(self.n):
                if self.rng.random() < float(self.cfg.mutation_prob):
                    dlam[i, j] -= float(self.cfg.mutation_erosion)

        self.lam = np.clip(self.lam + dlam, self.cfg.lam_min, self.cfg.lam_max)

        # 4) optional interventions (if you attach any)
        for itv in getattr(self, "interventions", []):
            # support both styles:
            # - itv.apply(grid, t)
            # - itv.apply(grid, ctx)  (ctx.t)
            if hasattr(itv, "apply"):
                try:
                    itv.apply(self, t)
                except TypeError:
                    class _Ctx:
                        def __init__(self, t: int):
                            self.t = t
                    itv.apply(self, _Ctx(t))

    # ----------------------------
    # Convenience metrics
    # ----------------------------
    def mean_gv(self) -> float:
        return float(self.gv_field().mean())

    def mean_lam(self) -> float:
        return float(self.lam_field().mean())


def run_demo() -> None:
    """
    CLI demo: python -m cancer_project.grid
    """
    cfg = GridConfig()
    env = Environment()
    grid = Grid(cfg, env)

    print("t,mean_gv,mean_lam")
    for t in range(cfg.steps + 1):
        print(f"{t},{grid.mean_gv():.6f},{grid.mean_lam():.6f}")
        grid.step(t)


# Back-compat alias (some scripts look for run_grid)
def run_grid() -> None:
    run_demo()


if __name__ == "__main__":
    run_demo()
