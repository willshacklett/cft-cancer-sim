from cancer_project import Environment, HealthyCell, CancerCell, gv_score

def test_cancer_accumulates_more_gv_than_healthy():
    env = Environment(toxins=0.2, oxygen=0.5)

    healthy = HealthyCell()
    cancer = CancerCell()

    for _ in range(30):
        healthy.step(env)
        cancer.step(env)

    gv_h = gv_score(healthy.atp, healthy.damage, healthy.arrest_steps, healthy.divisions)
    gv_c = gv_score(cancer.atp, cancer.damage, cancer.arrest_steps, cancer.divisions)

    assert gv_c > gv_h
