import matplotlib
matplotlib.use("Agg")  # headless for CI / Codespaces

import matplotlib.pyplot as plt
from cancer_project.grid import run_grid


def main():
    # Run grid simulation
    history = run_grid(steps=60)

    # Unpack
    t = [row[0] for row in history]
    mean_gv = [row[1] for row in history]
    mean_lambda = [row[2] for row in history]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(t, mean_gv, label="Mean GV", linewidth=2)
    plt.plot(t, mean_lambda, label="Mean λ (constraint tightness)", linewidth=2)

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Multicell Emergence: Mean GV vs Constraint Tightness")
    plt.legend()
    plt.grid(True)

    # Save artifact
    out_path = "assets/gv_grid_mean.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)

    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
