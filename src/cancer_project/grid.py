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
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np

# Import single-cell components (must exist in cancer_project)
try:
    from cancer_project import Environment, HealthyCell, CancerCell
except Exception as e:
    raise ImportError(
        "Could not import Environment / HealthyCell / CancerCell from cancer_project. "
        "Make sure they exist and are exported in cancer_project/__init__.py"
    ) from e

# Optional intervention API (safe if you don't use it)
try:
    from cancer_project.intervention import Intervention, InterventionContext, NoOpIntervention
except Exception:
    Intervention = None  # type: ignore
    InterventionContext = None  # type: ignore

    class NoOpIntervention:  # fallback
        def apply(self, grid: "Grid", ctx: object) -> None:
            return


@dataclass
class GridConfig:
    # grid
    n: int = 20
    steps: int = 60

    # initial conditions
    seed_cancer_prob: float = 0.02
    init_lambda: float = 1.0
    init_gv: float = 0.0

    # coupling dynamics (lambda field)
    neighbor_strength: float = 0.015  # how strongly neighbors affect lam
    healthy_reinforce: float = 1.0    # multiplier for healthy sites
    cancer_erode: float = 1.6         # multiplier for cancer sites

    # mutation / degradation
    mutation_prob: float = 0.004
    mutation_drop: float = 0.10

    # clamps
    lambda_min: float = 0.05
    lambda_max: float = 2.0

    # GV update
    base_strain: float = 0.003
    toxin_weight: float = 0.018
    oxygen_weight: float = 0.006
    nutrient_weight: float = 0.004
    cancer_bonus: float = 0.006

    feedback_strength: float = 0.55  # multiplies lam * gv for damping


class Grid:
    """
    Grid holds:
    - cells: HealthyCell or CancerCell objects (type defines coupling sign)
    - gv: numpy array (n,n)
    - lam: numpy array (n,n)  (constraint tightness)
    """

    def __init__(
        self,
        cfg: GridConfig,
        env: Environment,
        intervention: Optional[object] = None,
        seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.env = env
        self.rng = random.Random(seed)

        self.cells = np.empty((cfg.n, cfg.n), dtype=object)
        for i in range(cfg.n):
            for j in range(cfg.n):
                if self.rng.random() < cfg.seed_cancer_prob:
                    self.cells[i, j] = CancerCell()
                else:
                    self.cells[i, j] = HealthyCell()

        self.gv = np.full((cfg.n, cfg.n), float(cfg.init_gv), dtype=float)
        self.lam = np.full((cfg.n, cfg.n), float(cfg.init_lambda), dtype=float)

        # intervention hook (defaults to NoOp)
        self.intervention = intervention if intervention is not None else NoOpIntervention()

        self.t = 0

    @property
    def mean_gv(self) -> float:
        return float(self.gv.mean())

    @property
    def mean_lambda(self) -> float:
        return float(self.lam.mean())

    def _neighbors4(self, i: int, j: int):
        n = self.cfg.n
        if i > 0:
            yield (i - 1, j)
        if i < n - 1:
            yield (i + 1, j)
        if j > 0:
            yield (i, j - 1)
        if j < n - 1:
            yield (i, j + 1)

    def step(self) -> None:
        cfg = self.cfg
        n = cfg.n

        # --- 1) update lambda via neighbor coupling + mutations
        lam_next = self.lam.copy()

        # Local coupling: each site pushes neighbor lambda up/down
        for i in range(n):
            for j in range(n):
                cell = self.cells[i, j]
                is_cancer = isinstance(cell, CancerCell)

                # healthy reinforces; cancer erodes
                sign = -1.0 if is_cancer else +1.0
                mult = cfg.cancer_erode if is_cancer else cfg.healthy_reinforce

                delta = sign * mult * cfg.neighbor_strength

                for (ni, nj) in self._neighbors4(i, j):
                    lam_next[ni, nj] += delta

                # rare mutation: drop lambda at the site itself
                if self.rng.random() < cfg.mutation_prob:
                    lam_next[i, j] -= cfg.mutation_drop

        # clamp lambda
        lam_next = np.clip(lam_next, cfg.lambda_min, cfg.lambda_max)
        self.lam = lam_next

        # --- 2) update GV (strain accumulation - damped by lambda feedback)
        # Treat Environment fields as scalars (0..1-ish). If yours differ, tweak weights.
        toxins = float(getattr(self.env, "toxins", 0.0))
        oxygen = float(getattr(self.env, "oxygen", 0.5))
        nutrients = float(getattr(self.env, "nutrients", 0.5))

        # baseline strain + environmental contributions
        strain = (
            cfg.base_strain
            + cfg.toxin_weight * toxins
            + cfg.oxygen_weight * max(0.0, 1.0 - oxygen)
            + cfg.nutrient_weight * max(0.0, 1.0 - nutrients)
        )

        gv_next = self.gv.copy()
        for i in range(n):
            for j in range(n):
                is_cancer = isinstance(self.cells[i, j], CancerCell)
                extra = cfg.cancer_bonus if is_cancer else 0.0

                # accumulate strain
                gv_next[i, j] += (strain + extra)

                # feedback damping proportional to lambda * gv
                gv_next[i, j] -= cfg.feedback_strength * self.lam[i, j] * gv_next[i, j]

                if gv_next[i, j] < 0.0:
                    gv_next[i, j] = 0.0

        self.gv = gv_next

        # --- 3) optional intervention hook (safe no-op by default)
        try:
            if InterventionContext is not None:
                self.intervention.apply(self, InterventionContext(t=self.t))
            else:
                self.intervention.apply(self, {"t": self.t})
        except Exception:
            # Keep model robust: interventions should never crash the sim by default
            pass

        self.t += 1

    def snapshot(self) -> Dict[str, Any]:
        """Convenient for plotting/scripts."""
        return {
            "t": self.t,
            "gv": self.gv.copy(),
            "lambda": self.lam.copy(),
        }


def run_grid(
    cfg: Optional[GridConfig] = None,
    env: Optional[Environment] = None,
    intervention: Optional[object] = None,
    seed: Optional[int] = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a grid sim and return 4 fields:

    (gv_t0, gv_t_end, lambda_t0, lambda_t_end)

    This signature is intentionally simple for plot scripts.
    """
    cfg = cfg or GridConfig()
    env = env or Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)

    grid = Grid(cfg, env, intervention=intervention, seed=seed)

    gv0 = grid.gv.copy()
    lam0 = grid.lam.copy()

    for _ in range(cfg.steps):
        grid.step()

    gvT = grid.gv.copy()
    lamT = grid.lam.copy()

    return gv0, gvT, lam0, lamT


def run_demo() -> None:
    """
    Prints a simple emergence signal:
    - mean GV over time
    - mean lambda over time
    """
    cfg = GridConfig()
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)
    grid = Grid(cfg, env, seed=0)

    print("t,mean_gv,mean_lambda")
    for _ in range(cfg.steps):
        grid.step()
        print(f"{grid.t},{grid.mean_gv:.6f},{grid.mean_lambda:.6f}")


if __name__ == "__main__":
    run_demo()
