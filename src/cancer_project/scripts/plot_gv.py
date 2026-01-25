import os

print(">>> plot_gv.py is executing <<<")

test_path = os.path.join(
    os.path.dirname(__file__),
    "EXECUTION_CONFIRMED.txt"
)

with open(test_path, "w") as f:
    f.write("This file proves plot_gv.py executed.\n")

print("Wrote:", test_path)
