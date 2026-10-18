"""
Step 5b: Does the result hold when boundary noise is more tightly concentrated?

The main run used scale = 1.0, where boundary noise is only mildly
concentrated (~27% of points flipped near the line vs ~8% far away).
This reruns the boundary condition at tighter scales (0.5, 0.25) for every
setup and seed in step5_experiment.py.

Only the boundary labels depend on scale: the training points, the test
points and the uniform-noise labels are identical for a given seed. So each
new boundary run is compared seed by seed against the uniform run already in
results/results.csv.

Writes results/scale_sweep.csv and results/scale_sweep_summary.txt.
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
from step5_experiment import CONFIGS, SEEDS, TARGET_P

SCALES = (0.5, 0.25)
NEAR, FAR = 0.5, 1.5  # distance bands for describing how concentrated the flips are


def concentration(data: dict, y: np.ndarray) -> tuple[float, float]:
    """Share of training points flipped within NEAR of the true line, and beyond FAR."""
    dist = np.abs(data["dist"])
    flipped = y != data["labels"]["clean"]
    return float(flipped[dist < NEAR].mean()), float(flipped[dist > FAR].mean())


def run_one(job: tuple) -> dict:
    (setup, model_type, n, d, width, lr, steps), seed, scale = job
    data = make_conditions(n, d, TARGET_P, scale, seed, exact=True)
    y = data["labels"]["boundary"]
    log_every = steps // 100
    if model_type == "linear":
        model = train_linear(data["X"], y, data["X_test"], data["y_test"], lr, steps, log_every)
    else:
        model = train_net(data["X"], y, data["X_test"], data["y_test"], width, lr, steps, log_every, seed=seed)
    metrics, _ = compute_metrics(model, data, y)
    near, far = concentration(data, y)
    return {"setup": setup, "n": n, "d": d, "width": width, "seed": seed, "scale": scale,
            "flip_near": near, "flip_far": far, **metrics}


def summarize(rows: list[dict]) -> str:
    with open("results/results.csv") as f:
        base = list(csv.DictReader(f))
    key = lambda r: (r["setup"], int(r["d"]), int(r["width"]), int(r["seed"]))
    test = {(key(r), r["condition"]): float(r["test_err"]) for r in base}

    # scale = 1.0 concentration, recomputed (no training needed)
    for setup, _, n, d, width, _, _ in CONFIGS:
        for seed in SEEDS:
            data = make_conditions(n, d, TARGET_P, 1.0, seed, exact=True)
            near, far = concentration(data, data["labels"]["boundary"])
            rows.append({"setup": setup, "d": d, "width": width, "seed": seed, "scale": 1.0,
                         "flip_near": near, "flip_far": far,
                         "test_err": test[((setup, d, width, seed), "boundary")]})

    lines = []
    for setup, _, n, d, width, _, _ in CONFIGS:
        size = width if setup == "net" else d
        label = f"{setup}, {'width' if setup == 'net' else 'd'}={size}"
        lines.append(f"\n=== {label} (n={n}, 10 seeds) ===")
        u = np.array([test[((setup, d, width, s), "uniform")] for s in SEEDS])
        lines.append(f"uniform                                   test_err {u.mean():.4f}")
        for scale in (1.0,) + SCALES:
            rs = sorted((r for r in rows if r["setup"] == setup and r["d"] == d
                         and r["width"] == width and r["scale"] == scale), key=lambda r: r["seed"])
            b = np.array([r["test_err"] for r in rs])
            near = np.mean([r["flip_near"] for r in rs])
            far = np.mean([r["flip_far"] for r in rs])
            diff = u - b
            se = diff.std(ddof=1) / np.sqrt(len(diff))
            lines.append(
                f"boundary scale={scale:<4}  flips near {near:5.1%} far {far:5.1%}  "
                f"test_err {b.mean():.4f}  uniform-boundary {diff.mean():+.4f} "
                f"(t={diff.mean() / se:+.2f}, uniform worse {int((diff > 0).sum())}/{len(diff)})"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    jobs = [(config, seed, scale) for config in CONFIGS for seed in SEEDS for scale in SCALES]
    jobs.sort(key=lambda j: -(j[0][2] * max(j[0][4], j[0][3]) * j[0][6]))

    rows = []
    with Pool(processes=max(1, os.cpu_count() - 1)) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, jobs), start=1):
            rows.append(row)
            print(f"finished {i}/{len(jobs)} jobs", flush=True)

    rows.sort(key=lambda r: (r["setup"], r["d"], r["width"], r["scale"], r["seed"]))
    with open("results/scale_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    with open("results/scale_sweep_summary.txt", "w") as f:
        f.write(summary + "\n")
    print(summary)
