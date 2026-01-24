from __future__ import annotations
import random
from typing import List, Dict, Any

from .env import Environment
from .healthy_cell import HealthyCell
from .gv import gv_score

def run_sim(steps: int = 60, seed: int = 42, env: Environment | None = None) -> List[Dict[str, Any]]:
    if env is None:
        env = Environment()
    rng = random.Random(seed)
    cell = HealthyCell()

    history: List[Dict[str, Any]] = []
    for t in range(steps):
        state = cell.step(env, rng=rng)
        state["t"] = t
        state["gv"] = round(gv_score(cell.atp, cell.damage, cell.arrest_steps, cell.divisions), 4)
        history.append(state)
        if not cell.alive:
            break
    return history
