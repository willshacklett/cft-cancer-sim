from __future__ import annotations
from dataclasses import dataclass, field
import random

from .healthy_cell import HealthyCell, HealthyCellParams
from .env import Environment
from .cell_base import Phase


@dataclass
class CancerCellParams(HealthyCellParams):
    """
    Cancer-specific parameter overrides.
    These represent constraint exploitation / failure.
    """
    # Cancer tolerates more damage before death
    apoptosis_damage_threshold: float = 95.0

    # Checkpoints effectively disabled
    checkpoint_damage_threshold: float = 1e9
    max_arrest_steps: int = 1_000_000

    # Genomic instability: weaker repair
    damage_repair_rate: float = 0.6

    # Growth bias
    base_progress_rate: float = 3.0
    division_atp_cost: float = 15.0


@dataclass
class CancerCell(HealthyCell):
    """
    CancerCell = HealthyCell with constraint bypass.
    """
    params: CancerCellParams = field(default_factory=CancerCellParams)

    def step(self, env: Environment, dt: float = 1.0, rng: random.Random | None = None):
        if rng is None:
            rng = random.Random()

        if not self.alive:
            return self.snapshot()

        # --- ATP production (often survives hypoxia better) ---
        hypoxia_boost = 0.15  # crude Warburg-like effect
        production = 6.0 * env.nutrients * (0.5 + 0.5 * env.oxygen + hypoxia_boost) * dt
        self.atp = min(self.params.atp_max, self.atp + production)

        # --- ATP spend (still costly, but biased toward division) ---
        phase_mult = {
            Phase.G1: 1.0,
            Phase.S:  1.3,
            Phase.G2: 1.1,
            Phase.M:  1.4,
            Phase.QUIESCENT: 0.8,
            Phase.APOPTOTIC: 0.0,
        }[self.phase]

        self.atp -= self.params.basal_atp_use * phase_mult * dt

        # --- Damage accumulation (higher instability) ---
        metabolic_stress = max(0.0, (self.params.atp_min_survival - self.atp) / self.params.atp_min_survival)
        damage_gain = (
            self.params.damage_from_metabolic_stress * metabolic_stress +
            self.params.damage_from_toxins * env.toxins * 1.3
        ) * dt

        # Replication damage more frequent
        if self.phase == Phase.S and rng.random() < 0.35:
            damage_gain += rng.uniform(2.0, 8.0)

        self.damage = min(self.params.damage_max, self.damage + damage_gain)

        # --- Weakened repair ---
        repair = self.params.damage_repair_rate * dt
        self.damage = max(0.0, self.damage - repair)

        # --- Apoptosis (much harder to trigger) ---
        if self.atp <= -10.0 or self.damage >= self.params.apoptosis_damage_threshold:
            self.alive = False
            self.phase = Phase.APOPTOTIC
            self.phase_progress = 0.0
            return self.snapshot()

        # --- NO CHECKPOINT ARREST ---
        # Cancer ignores damage-based halts completely

        # --- Aggressive cycle progression ---
        gf = max(0.0, min(1.0, env.growth_factors))
        progress_rate = self.params.base_progress_rate * (0.5 + 0.5 * gf)
        self.phase_progress += progress_rate * dt

        if self.phase in self.params.phase_progress_required and \
           self.phase_progress >= self.params.phase_progress_required[self.phase]:
            self._advance_phase()

        return self.snapshot()
