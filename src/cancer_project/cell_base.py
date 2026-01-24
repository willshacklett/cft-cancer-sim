from dataclasses import dataclass
from enum import Enum

class Phase(str, Enum):
    G1 = "G1"
    S = "S"
    G2 = "G2"
    M = "M"
    QUIESCENT = "G0"
    APOPTOTIC = "APOPTOTIC"

@dataclass
class CellBase:
    atp: float = 80.0
    damage: float = 0.0
    phase: Phase = Phase.G1
    phase_progress: float = 0.0
    arrest_steps: int = 0
    alive: bool = True
    divisions: int = 0
