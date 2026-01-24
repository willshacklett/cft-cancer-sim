from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Dict, Any

from .env import Environment
from .cell_base import CellBase, Phase

@dataclass
class HealthyCellParams:
    atp_max: float = 100.0
    atp_min_survival: float = 10.0
    basal_atp_use: float = 1.5
    division_atp_cost: float = 25.0

    damage_max: float = 100.0
    damage_repair_rate: float = 2.0
    damage_from_metabolic_stress: float = 1.2
    damage_from_toxins: float = 8.0

    checkpoint_damage_threshold: float = 25.0
    apoptosis_damage_threshold: float = 70.0
    max_arrest_steps: int = 15

    phase_progress_required: Dict[Phase, float] = field(default_factory=lambda: {
        Phase.G1: 20.0,
        Phase.S:  25.0,
        Phase.G2: 15.0,
        Phase.M:  10.0,
    })

    base_progress_rate: float = 2.0

@dataclass
class HealthyCell(CellBase):
    params: HealthyCellParams = field(default_factory=HealthyCellParams)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "alive": self.alive,
            "phase": self.phase.value,
            "phase_progress": round(self.phase_progress, 3),
            "atp": round(self.atp, 3),
            "damage": round(self.damage, 3),
            "arrest_steps": self.arrest_steps,
            "divisions": self.divisions,
        }

    def step(self, env: Environment, dt: float = 1.0, rng: random.Random | None = None) -> Dict[str, Any]:
        if rng is None:
            rng = random.Random()

        if not self.alive:
            return self.snapshot()

        # ATP production
        production = 6.0 * env.nutrients * (0.5 + 0.5 * env.oxygen) * dt
        self.atp = min(self.params.atp_max, self.atp + production)

        # ATP spend
        phase_mult = {Phase.G1:1.0, Phase.S:1.4, Phase.G2:1.1, Phase.M:1.6, Phase.QUIESCENT:0.6, Phase.APOPTOTIC:0.0}[self.phase]
        self.atp -= self.params.basal_atp_use * phase_mult * dt

        # Damage gain
        metabolic_stress = max(0.0, (self.params.atp_min_survival - self.atp) / self.params.atp_min_survival)
        damage_gain = (self.params.damage_from_metabolic_stress * metabolic_stress + self.params.damage_from_toxins * env.toxins) * dt

        if self.phase == Phase.S and rng.random() < min(0.25, 0.05 + 0.1*env.toxins + 0.1*metabolic_stress):
            damage_gain += rng.uniform(1.0, 6.0)

        self.damage = min(self.params.damage_max, self.damage + damage_gain)

        # Repair
        energy_factor = max(0.0, min(1.0, self.atp / self.params.atp_max))
        repair = self.params.damage_repair_rate * (0.4 + 0.6 * energy_factor) * dt
        self.damage = max(0.0, self.damage - repair)

        # Death checks
        if self.atp <= 0.0 or self.damage >= self.params.apoptosis_damage_threshold:
            self.alive = False
            self.phase = Phase.APOPTOTIC
            self.phase_progress = 0.0
            return self.snapshot()

        # Checkpoint arrest
        if self.damage >= self.params.checkpoint_damage_threshold:
            self.arrest_steps += 1
            if self.arrest_steps >= self.params.max_arrest_steps:
                self.alive = False
                self.phase = Phase.APOPTOTIC
            return self.snapshot()
        else:
            self.arrest_steps = max(0, self.arrest_steps - 1)

        # Progress cell cycle
        gf = max(0.0, min(1.0, env.growth_factors))
        if self.phase == Phase.G1 and gf < 0.15:
            self.phase = Phase.QUIESCENT
            self.phase_progress = 0.0
            return self.snapshot()

        progress_rate = self.params.base_progress_rate * (0.3 + 0.7 * gf) * (0.4 + 0.6 * energy_factor)
        self.phase_progress += progress_rate * dt

        if self.phase in self.params.phase_progress_required and self.phase_progress >= self.params.phase_progress_required[self.phase]:
            self._advance_phase()

        if self.phase == Phase.QUIESCENT and gf >= 0.2 and self.atp > self.params.atp_min_survival:
            self.phase = Phase.G1
            self.phase_progress = 0.0

        return self.snapshot()

    def _advance_phase(self) -> None:
        self.phase_progress = 0.0
        if self.phase == Phase.G1: self.phase = Phase.S
        elif self.phase == Phase.S: self.phase = Phase.G2
        elif self.phase == Phase.G2: self.phase = Phase.M
        elif self.phase == Phase.M:
            if self.atp >= self.params.division_atp_cost:
                self.atp -= self.params.division_atp_cost
                self.divisions += 1
                self.phase = Phase.G1
            else:
                self.arrest_steps += 1
