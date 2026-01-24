from cancer_project import Environment, HealthyCell, gv_score

def test_healthy_cell_runs_and_gv_in_range():
    env = Environment()
    cell = HealthyCell()

    for _ in range(20):
        cell.step(env)

    score = gv_score(cell.atp, cell.damage, cell.arrest_steps, cell.divisions)
    assert 0.0 <= score <= 2.0  # toy bound; should usually be <= 1.0
