# Cancer Project

A toy cancer dynamics simulator built on **GV (scalar strain / risk)** and **CFT-inspired constraint coupling**.

This project is intentionally simple: it demonstrates how **constraint failure** drives divergence between healthy and cancerous systems over time.

---

## GV Trajectory: Healthy vs Cancer

![GV Healthy vs Cancer](assets/gv_healthy_vs_cancer.png)

---

## What this is

- A **baseline HealthyCell model**
- A **CancerCell model** that exploits weakened constraints
- A **GV (God Variable) score** that rises as strain accumulates  
  (energy loss, damage, arrest failures, uncontrolled division)
- A reproducible simulation that **executes, plots, and saves artifacts**

This is not a medical claim.  
It is a **systems-level demonstration**.

---

## Project structure

```text
cft-cancer-sim/
├── README.md
├── pyproject.toml
├── poetry.lock
├── assets/
│   └── gv_healthy_vs_cancer.png
├── src/
│   └── cancer_project/
│       ├── __init__.py
│       ├── env.py
│       ├── gv.py
│       ├── cell_base.py
│       ├── healthy_cell.py
│       ├── cancer_cell.py
│       ├── sim.py
│       └── scripts/
│           ├── __init__.py
│           └── plot_gv.py
└── tests/
