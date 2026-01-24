from dataclasses import dataclass

@dataclass
class Environment:
    nutrients: float = 0.8
    oxygen: float = 0.8
    toxins: float = 0.05
    growth_factors: float = 0.6
