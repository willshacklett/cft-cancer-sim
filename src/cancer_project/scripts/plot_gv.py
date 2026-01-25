import os
import matplotlib.pyplot as plt

from cancer_project import (
    Environment,
    HealthyCell,
    CancerCell,
    gv_score,
)


def run(cell, env, steps=50):
    gv_values = []

    for _ in range(steps):
        cell.step(env)

        gv_values.append(
            gv_score(
                cell.atp,
                cell.damage,
                cell.arrest_steps,
                cell.divisions,
            )
        )

        if not cell.alive:
            break

    return gv_values


if __name__ == "__main__":
    # ----------------------------
    # Environment stress scenario
    # ----------------------------
    env = Environment(
        toxins=0.2,
        oxygen=0.5,
        nutrients=0.7,
    )

    healthy = HealthyCell()
    cancer = CancerCell()

    gv_healthy = run(healthy, env)
    gv_cancer = run(cancer, env)

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(gv_healthy, label="Healthy Cell", linewidth=2)
    plt.plot(gv_cancer, label="Cancer Cell", linewidth=2)

    plt.xlabel("Time step")
    plt.ylabel("GV (strain / risk)")
    plt.title("GV Trajectory: Healthy vs Cancer")
    plt.legend()
    plt.grid(True)

    # ----------------------------
    # Save plot next to this file
    # ----------------------------
    output_path = os.path.join(
        os.path.dirname(__file__),
        "gv_healthy_vs_cancer.png"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot to: {output_path}")
