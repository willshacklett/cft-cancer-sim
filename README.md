# Cancer Project

A toy cancer dynamics simulator built on **GV (scalar strain/risk)** and **CFT-inspired constraint coupling**.

## What this is
- A **baseline healthy cell model**
- A **GV score** that rises when constraints fail (energy, repair, checkpoints)
- A simulator that will later add **CancerCell** as “constraint exploitation”

## Run (local)
```bash
python -c "from cancer_project import run_sim; print(run_sim(steps=30)[:5])"
