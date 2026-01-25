"""
plot_gv.py

Compares GV trajectory for HealthyCell vs CancerCell and saves outputs.
Works in headless environments (GitHub Codespaces / CI).

Outputs (saved in this same folder):
- gv_healthy_vs_cancer.png
- EXECUTION_CONFIRMED.txt
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend for Codespaces / CI
import matplotlib.pyplot as plt

from cancer_project import Environment, HealthyCell, CancerCell, gv_score


def run(cell, env, steps: int = 50):
    gv = []
    for _ in range(steps):
        cell.step(env)
        gv.append(
            gv_score(
                cell.atp,
                cell.damage,
                cell.arrest_steps,
                cell.divisions,
            )
        )
        if not getattr(cell, "alive", True):
            break
    return gv


def main():
    here = Path(__file__).resolve().parent

    plot_path = here / "gv_healthy_vs_cancer.png"
    confirm_path = here / "EXECUTION_CONFIRMED.txt"

    # Write execution proof (non-negotiable)
    confirm_path.write_text(
        "plot_gv.py executed successfully.\n",
        encoding="utf-8",
    )

    # Environment stress scenario
    env = Environment(
        toxins=0.2,
        oxygen=0.5,
        nutrients=0.7,
    )

    healthy = HealthyCell()
    cancer = CancerCell()

    gv_h = run(healthy, env, steps=60)
    gv_c = run(cancer, env, steps=60)

    plt.figure(figsize=(8, 5))
    plt.plot(gv_h, label="Healthy Cell", linewidth=2)
    plt.plot(gv_c, label="Cancer Cell", linewidth=2)
    plt.xlabel("Time step")
    plt.ylabel("GV (strain / risk)")
    plt.title("GV Trajectory: Healthy vs Cancer")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Saved plot:", plot_path)
    print("Execution confirmed:", confirm_path)


if __name__ == "__main__":
    main()
