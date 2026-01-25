"""
Minimal multicell emergence model (systems demo only).

A 2D grid of sites. Each site has:
- a single-cell object (HealthyCell / CancerCell) reused from the single-cell model
- gv: accumulated "strain / risk" scalar (computed from cell state)
- lam: local constraint tightness (feedback strength), stored at the grid level

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

# --- single-cell components must exist in cancer_project ---
try:
    from cancer_project import Environment, HealthyCell, CancerCell  # type: ignore
except Exception as e:
    raise ImportError(
        "Could not import Environment / HealthyCell / CancerCell from cancer_project. "
        "Make sure they exist and are exported in cancer_project/__init__.py"
    ) from e

# GV scoring function (if present)
try:
    from cancer_project.gv import gv_score  # type: ignore
except Exception:
    gv_score = None  # we'll fall back to a simple proxy if missing

# Optional intervention API (safe if you don't use it)
try:
    from cancer_project.intervention import (  # type: ignore
        Intervention,
        InterventionContext,
        NoOpIntervention,
    )
except Exception:
    Intervention = None  # type: ignore
    InterventionContext = None  # type: ignore

    class NoOpIntervention:  # type: ignore
        def apply(self, grid: "Grid", ctx: Any) -> None:
            return


@dataclass
class GridConfig:
    n: int = 20
    steps: int = 60

    # init
    init_cancer_prob: float = 0.03
    seed: int = 7

    # constraint coupling (lambda dynamics)
    lam_init: float = 1.0
    lam_min: float = 0.05
    lam_max: float = 2.0

    healthy_reinforce: float = 0.010
    cancer_erode: float = 0.018

    neighbor_strength: float = 0.12  # neighbor averaging weight
    mutation_rate: float = 0.002     # chance per site per step
    mutation_drop: float = 0.25      # lam *= (1 - mutation_drop)

    # gv smoothing
    gv_smooth: float = 0.15  # EMA mix for gv field (0=no smooth)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _is_cancer(cell: Any) -> bool:
    # robust: class name check avoids tight coupling
    return cell.__class__.__name__.lower().startswith("cancer")


def _safe_get(obj: Any, name: str, default: float = 0.0) -> float:
    v = getattr(obj, name, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _compute_gv(cell: Any) -> float:
    """
    Prefer gv_score() if available; otherwise compute a stable proxy.
    """
    # pull common state variables defensively
    atp = _safe_get(cell, "atp", 0.0)
    damage = _safe_get(cell, "damage", 0.0)
    arrest_steps = int(_safe_get(cell, "arrest_steps", 0.0))
    divisions = int(_safe_get(cell, "divisions", 0.0))

    if gv_score is not None:
        try:
            return float(gv_score(atp=atp, damage=damage, arrest_steps=arrest_steps, divisions=divisions))
        except Exception:
            pass

    # fallback proxy: higher damage + lower ATP => higher GV (risk/strain)
    # bounded to keep plots sane
    proxy = (damage * 0.08) + (1.0 / (1.0 + max(atp, 0.0)))
    return float(_clamp(proxy, 0.0, 5.0))


class Grid:
    def __init__(
        self,
        cfg: GridConfig,
        env: Optional["Environment"] = None,
        intervention: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.env = env if env is not None else Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)

        # intervention hook
        self.intervention = intervention if intervention is not None else NoOpIntervention()

        # cells + fields
        self.cells = np.empty((cfg.n, cfg.n), dtype=object)
        self.lam = np.full((cfg.n, cfg.n), float(cfg.lam_init), dtype=float)
        self.gv = np.zeros((cfg.n, cfg.n), dtype=float)

        self._init_cells()

    def _init_cells(self) -> None:
        for i in range(self.cfg.n):
            for j in range(self.cfg.n):
                if self.rng.random() < self.cfg.init_cancer_prob:
                    self.cells[i, j] = self._make_cell(cancer=True)
                else:
                    self.cells[i, j] = self._make_cell(cancer=False)
        self._refresh_gv()

    def _make_cell(self, cancer: bool) -> Any:
        """
        IMPORTANT: don't pass positional args.
        Some of your classes are dataclasses where the first field is 'atp',
        so passing env positionally will corrupt state (env becomes atp).
        """
        if cancer:
            try:
                return CancerCell()  # type: ignore
            except TypeError:
                return CancerCell  # pragma: no cover
        else:
            try:
                return HealthyCell()  # type: ignore
            except TypeError:
                return HealthyCell  # pragma: no cover

    def neighbors(self, i: int, j: int) -> Tuple[Tuple[int, int], ...]:
        n = self.cfg.n
        coords = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < n and 0 <= jj < n:
                coords.append((ii, jj))
        return tuple(coords)

    def step(self, t: int = 0) -> None:
        """
        One grid timestep:
        1) apply intervention (optional)
        2) update single cells
        3) update lam coupling field
        4) refresh gv field
        """
        # (1) intervention
        try:
            ctx = InterventionContext(t=t) if InterventionContext is not None else {"t": t}
            self.intervention.apply(self, ctx)
        except Exception:
            # keep demo robust
            pass

        # (2) update cells using env
        for i in range(self.cfg.n):
            for j in range(self.cfg.n):
                c = self.cells[i, j]
                # HealthyCell.step(env, dt=..., rng=...)
                # use keyword args so signature changes won't explode
                try:
                    c.step(env=self.env, dt=1.0, rng=self.rng)  # type: ignore
                except TypeError:
                    # older signature variants
                    try:
                        c.step(self.env)  # type: ignore
                    except Exception:
                        pass

        # (3) lambda coupling: neighbor influence + cell-type action + rare mutation drops
        new_lam = self.lam.copy()
        for i in range(self.cfg.n):
            for j in range(self.cfg.n):
                c = self.cells[i, j]
                nbrs = self.neighbors(i, j)
                if nbrs:
                    nbr_mean = float(np.mean([self.lam[ii, jj] for (ii, jj) in nbrs]))
                    # pull toward neighbor mean
                    new_lam[i, j] = (1.0 - self.cfg.neighbor_strength) * new_lam[i, j] + self.cfg.neighbor_strength * nbr_mean

                # cell-type action
                if _is_cancer(c):
                    new_lam[i, j] -= self.cfg.cancer_erode
                else:
                    new_lam[i, j] += self.cfg.healthy_reinforce

                # stochastic mutation (constraint relaxation)
                if self.rng.random() < self.cfg.mutation_rate:
                    new_lam[i, j] *= (1.0 - self.cfg.mutation_drop)

                new_lam[i, j] = _clamp(new_lam[i, j], self.cfg.lam_min, self.cfg.lam_max)

        self.lam = new_lam

        # (4) refresh gv field (smoothed)
        self._refresh_gv()

    def _refresh_gv(self) -> None:
        g_new = np.zeros_like(self.gv)
        for i in range(self.cfg.n):
            for j in range(self.cfg.n):
                c = self.cells[i, j]
                g_new[i, j] = _compute_gv(c)

        a = float(self.cfg.gv_smooth)
        if a <= 0.0:
            self.gv = g_new
        else:
            self.gv = (1.0 - a) * self.gv + a * g_new

    # --- Public API expected by scripts/README ---

    def gv_field(self) -> np.ndarray:
        return self.gv.copy()

    def lam_field(self) -> np.ndarray:
        return self.lam.copy()

    def mean_gv(self) -> float:
        return float(np.mean(self.gv))

    def mean_lambda(self) -> float:
        return float(np.mean(self.lam))


def run_demo() -> None:
    """
    Prints a simple emergence signal:
    - mean GV over time
    - mean lambda over time
    """
    cfg = GridConfig()
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)
    grid = Grid(cfg, env=env)

    print("t,mean_gv,mean_lambda")
    for t in range(cfg.steps + 1):
        if t > 0:
            grid.step(t=t)
        print(f"{t},{grid.mean_gv():.6f},{grid.mean_lambda():.6f}")


# Backwards-compatible alias for scripts expecting run_grid()
run_grid = run_demo


if __name__ == "__main__":
    run_demo()
