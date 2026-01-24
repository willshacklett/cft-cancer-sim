def gv_score(atp: float, damage: float, arrest_steps: int, divisions: int) -> float:
    """
    GV = scalar strain/risk score (toy).
    Higher means more unstable / higher survivability risk.

    Components:
      - damage drives risk strongly
      - low ATP increases risk
      - checkpoint arrest indicates system strain
      - divisions add mild “wear” (optional)
    """
    # clamp-ish behavior without importing numpy
    low_energy = max(0.0, 1.0 - (atp / 100.0))
    return (
        0.65 * (damage / 100.0) +
        0.20 * low_energy +
        0.12 * min(1.0, arrest_steps / 15.0) +
        0.03 * min(1.0, divisions / 10.0)
    )
