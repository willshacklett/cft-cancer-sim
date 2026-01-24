import matplotlib.pyplot as plt

from cancer_project import Environment, HealthyCell, CancerCell, gv_score

def run(cell, env, steps=50):
    gv = []
    for _ in range(steps):
        cell.step(env)
        gv.append(
            gv_score(cell.atp, cell.damage, cell.arrest_steps, cell.divisions)
        )
        if not cell.alive:
            break
    return gv

if __name__ == "__main__":
    env = Environment(toxins=0.2, oxygen=0.5, nutrients=0.7)

    healthy = HealthyCell()
    cancer = CancerCell()

    gv_h = run(healthy, env)
    gv_c = run(cancer, env)

    plt.plot(gv_h, label="Healthy Cell", linewidth=2)
    plt.plot(gv_c, label="Cancer Cell", linewidth=2)

    plt.xlabel("Time step")
    plt.ylabel("GV (strain / risk)")
    plt.title("GV Trajectory: Healthy vs Cancer")
    plt.legend()
    plt.grid(True)

    plt.show()
