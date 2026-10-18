"""
Step 5: Run the full experiment across seeds and model sizes.

Three setups, each run on clean / uniform-noise / boundary-noise labels:

- "net":          one-hidden-layer net, n=200, d=10, widths 128 / 512 / 1024
- "linear":       linear model with more dimensions than points, n=200, d=400 / 800
- "linear_under": linear model with far more points than dimensions,
                  n=2000, d=10 (cannot fit the noise; included as a contrast)

Both noise conditions flip exactly 20% of the training labels in every run.
Results go to results/results.csv (one row per run) and results/arrays.npz
(training curves and margins). Plots are made by step5_plots.py.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
from multiprocessing import Pool

import numpy as np

from step3_training import make_conditions, train_linear, train_net
from step4_metrics import compute_metrics

TARGET_P = 0.2
SCALE = 1.0
SEEDS = range(10)
CONDITIONS = ("clean", "uniform", "boundary")

# (setup, model, n, d, width, lr, steps)
CONFIGS = [
    ("net", "net", 200, 10, 128, 0.5, 30000),
    ("net", "net", 200, 10, 512, 0.5, 30000),
    ("net", "net", 200, 10, 1024, 0.5, 30000),
    ("linear", "linear", 200, 400, 0, 0.5, 20000),
    ("linear", "linear", 200, 800, 0, 0.5, 20000),
    ("linear_under", "linear", 2000, 10, 0, 0.5, 5000),
]


def run_one(job: tuple) -> list[tuple[dict, dict]]:
    """Train one config on one seed under all three label conditions."""
    (setup, model_type, n, d, width, lr, steps), seed = job
    data = make_conditions(n, d, TARGET_P, SCALE, seed, exact=True)
    log_every = steps // 100
    out = []
    for condition in CONDITIONS:
        y = data["labels"][condition]
        if model_type == "linear":
            model = train_linear(data["X"], y, data["X_test"], data["y_test"], lr, steps, log_every)
        else:
            model = train_net(data["X"], y, data["X_test"], data["y_test"], width, lr, steps, log_every, seed=seed)
        metrics, margins = compute_metrics(model, data, y)
        row = {"setup": setup, "n": n, "d": d, "width": width, "seed": seed, "condition": condition, **metrics}
        arrays = {"margins": margins, "flipped": y != data["labels"]["clean"], **model["curves"]}
        out.append((row, arrays))
    return out


if __name__ == "__main__":
    jobs = [(config, seed) for config in CONFIGS for seed in SEEDS]
    # biggest jobs first so the pool stays busy
    jobs.sort(key=lambda j: -(j[0][2] * max(j[0][4], j[0][3]) * j[0][6]))

    rows, saved = [], {}
    with Pool(processes=max(1, os.cpu_count() - 1)) as pool:
        for i, result in enumerate(pool.imap_unordered(run_one, jobs), start=1):
            for row, arrays in result:
                rows.append(row)
                size = row["width"] if row["setup"] == "net" else row["d"]
                key = f"{row['setup']}|{size}|{row['condition']}|{row['seed']}"
                for name, arr in arrays.items():
                    saved[f"{key}|{name}"] = arr
            print(f"finished {i}/{len(jobs)} jobs", flush=True)

    rows.sort(key=lambda r: (r["setup"], r["d"], r["width"], r["seed"], CONDITIONS.index(r["condition"])))
    os.makedirs("results", exist_ok=True)
    with open("results/results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed("results/arrays.npz", **saved)
    print("Saved results/results.csv and results/arrays.npz")
