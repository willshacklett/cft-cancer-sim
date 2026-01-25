import os
import matplotlib
matplotlib.use("Agg")  # headless safe (Codespaces/GitHub Actions)

import matplotlib.pyplot as plt

from cancer_project.grid import GridConfig, Grid
from cancer_project import Environment


def main() -> None:
    # Ensure assets exists
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
    assets_dir = os.path.join(repo_root, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    cfg = GridConfig(n=20, steps=60, init_cancer_prob=0.03, seed=7)
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)

    grid = Grid(cfg, env)

    gv0 = grid.gv_field()
    lam0 = grid.lambda_field()

    for _ in range(cfg.steps):
        grid.step()

    gv1 = grid.gv_field()
    lam1 = grid.lambda_field()

    out_path = os.path.join(assets_dir, "gv_grid_heatmap.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].imshow(gv0)
    axes[0, 0].set_title("GV field (t=0)")
    axes[0, 1].imshow(gv1)
    axes[0, 1].set_title(f"GV field (t={cfg.steps})")

    axes[1, 0].imshow(lam0)
    axes[1, 0].set_title("λ field (t=0)")
    axes[1, 1].imshow(lam1)
    axes[1, 1].set_title(f"λ field (t={cfg.steps})")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Minimal Multicell Emergence: GV + Constraint Tightness (λ)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
