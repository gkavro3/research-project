"""
Step 5 (continued): Summary table and figures from the saved results.

Reads results/results.csv and results/arrays.npz (made by step5_experiment.py)
and writes:

- results/summary.txt           mean +/- standard error per setup and condition,
                                plus the paired uniform-vs-boundary comparison
- results/fig1_test_error.png   test error by label condition, main setups
- results/fig2_training.png     test error and weight norm during training (net)
- results/fig3_margins.png      training-point margin distributions
- results/fig4_model_size.png   test error as model size changes
"""

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONDITIONS = ("clean", "uniform", "boundary")
LABELS = {"clean": "Clean labels", "uniform": "Uniform noise", "boundary": "Boundary noise"}
COLORS = {"clean": "#898781", "uniform": "#2a78d6", "boundary": "#eb6834"}
MARKERS = {"clean": "s", "uniform": "o", "boundary": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

# (setup, size) -> panel title; size is width for the net, d for linear
MAIN = {
    ("net", 512): "Shallow net (n=200, d=10, width 512)",
    ("linear", 400): "Linear, d > n (n=200, d=400)",
    ("linear_under", 10): "Linear, n > d (n=2000, d=10)",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": MUTED,
    "axes.titlesize": 9.5,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_rows() -> list[dict]:
    with open("results/results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k in ("setup", "condition"):
                continue
            r[k] = int(v) if k in ("n", "d", "width", "seed", "interp_step") else float(v)
        r["size"] = r["width"] if r["setup"] == "net" else r["d"]
    return rows


def group(rows: list[dict], field: str) -> dict:
    """{(setup, size): {condition: array of `field` ordered by seed}}"""
    out = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: r["seed"]):
        out[(r["setup"], r["size"])][r["condition"]].append(r[field])
    return {k: {c: np.array(v) for c, v in d.items()} for k, d in out.items()}


def mean_se(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))


def write_summary(rows: list[dict]) -> str:
    fields = ("train_err", "test_err", "test_err_near", "test_err_far", "min_margin", "norm")
    grouped = {f: group(rows, f) for f in fields + ("interp_step", "cosine", "flip_rate")}
    lines = []
    for key in sorted(grouped["test_err"]):
        setup, size = key
        n_seeds = len(grouped["test_err"][key]["clean"])
        lines.append(f"\n=== {setup}, size={size} ({n_seeds} seeds) ===")
        lines.append(f"{'condition':9s} {'flip':>6s} {'interp':>7s} " + " ".join(f"{f:>19s}" for f in fields))
        for c in CONDITIONS:
            interp = np.mean(grouped["interp_step"][key][c] >= 0)
            cells = []
            for f in fields:
                m, se = mean_se(grouped[f][key][c])
                cells.append(f"{m:10.4f} +/-{se:6.4f}")
            lines.append(f"{c:9s} {np.mean(grouped['flip_rate'][key][c]):6.3f} {interp:7.0%} " + " ".join(cells))
        if setup != "net":
            cos = " ".join(f"{c}={np.mean(grouped['cosine'][key][c]):.3f}" for c in CONDITIONS)
            lines.append(f"cosine(learned w, true w): {cos}")
        diff = grouped["test_err"][key]["uniform"] - grouped["test_err"][key]["boundary"]
        m, se = mean_se(diff)
        lines.append(
            f"paired test_err (uniform - boundary): mean={m:+.4f}, SE={se:.4f}, "
            f"t={m / se:+.2f}, uniform worse in {int(np.sum(diff > 0))}/{len(diff)} seeds"
        )
    text = "\n".join(lines)
    with open("results/summary.txt", "w") as f:
        f.write(text + "\n")
    return text


def fig_test_error(rows: list[dict]) -> None:
    test = group(rows, "test_err")
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    rng = np.random.default_rng(0)
    for ax, (key, title) in zip(axes, MAIN.items()):
        for i, c in enumerate(CONDITIONS):
            vals = 100 * test[key][c]
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(i + jitter, vals, s=14, color=COLORS[c], alpha=0.35, linewidths=0)
            m, se = mean_se(vals)
            ax.errorbar(i, m, yerr=se, fmt=MARKERS[c], color=COLORS[c], markersize=7,
                        markeredgecolor="white", markeredgewidth=1.2, capsize=3, elinewidth=1.2)
            ax.annotate(f"{m:.1f}%", (i, m), xytext=(12, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=8.5)
        ax.set_xticks(range(3), [LABELS[c].replace(" ", "\n") for c in CONDITIONS])
        ax.set_xlim(-0.5, 2.7)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Test error on clean labels (%)")
    fig.tight_layout()
    fig.savefig("results/fig1_test_error.png")
    plt.close(fig)


def fig_training(arrays, n_seeds: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    panels = (("test_err", "Test error on clean labels (%)", 100.0), ("norm", "Weight norm", 1.0))
    for ax, (name, ylabel, mult) in zip(axes, panels):
        for c in CONDITIONS:
            steps = arrays[f"net|512|{c}|0|step"]
            runs = mult * np.stack([arrays[f"net|512|{c}|{s}|{name}"] for s in range(n_seeds)])
            mean = runs.mean(axis=0)
            se = runs.std(axis=0, ddof=1) / np.sqrt(n_seeds)
            ax.plot(steps, mean, color=COLORS[c], linewidth=2, label=LABELS[c])
            ax.fill_between(steps, mean - se, mean + se, color=COLORS[c], alpha=0.15, linewidth=0)
        ax.set_xlabel("Gradient descent step")
        ax.set_ylabel(ylabel)
    axes[0].set_title("Shallow net (n=200, d=10, width 512)", loc="left")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("results/fig2_training.png")
    plt.close(fig)


def fig_margins(arrays, n_seeds: int) -> None:
    panels = (("net|512", "Shallow net (width 512)"), ("linear|400", "Linear, d > n (d=400)"))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, (prefix, title) in zip(axes, panels):
        pooled = {c: np.concatenate([arrays[f"{prefix}|{c}|{s}|margins"] for s in range(n_seeds)])
                  for c in CONDITIONS}
        hi = np.percentile(np.concatenate(list(pooled.values())), 99)
        bins = np.linspace(0, hi, 41)
        for c in CONDITIONS:
            ax.hist(pooled[c], bins=bins, histtype="step", color=COLORS[c], linewidth=1.6,
                    density=True, label=LABELS[c])
        ax.set_xlabel("Normalized margin of training point")
        ax.set_title(title, loc="left")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 3))
    axes[0].set_ylabel("Density")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("results/fig3_margins.png")
    plt.close(fig)


def fig_model_size(rows: list[dict]) -> None:
    test = group(rows, "test_err")
    panels = (("net", "Hidden width", "Shallow net (n=200, d=10)"),
              ("linear", "Input dimension d", "Linear, d > n (n=200)"))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, (setup, xlabel, title) in zip(axes, panels):
        sizes = sorted(size for (s, size) in test if s == setup)
        for c in CONDITIONS:
            stats = [mean_se(100 * test[(setup, size)][c]) for size in sizes]
            ax.errorbar(range(len(sizes)), [m for m, _ in stats], yerr=[se for _, se in stats],
                        fmt=MARKERS[c] + "-", color=COLORS[c], linewidth=2, markersize=7,
                        markeredgecolor="white", markeredgewidth=1.2, capsize=3, label=LABELS[c])
        ax.set_xticks(range(len(sizes)), [str(s) for s in sizes])
        ax.set_xlim(-0.3, len(sizes) - 0.7)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("Test error on clean labels (%)")
    axes[0].legend(loc="center left")
    fig.tight_layout()
    fig.savefig("results/fig4_model_size.png")
    plt.close(fig)


if __name__ == "__main__":
    rows = load_rows()
    arrays = np.load("results/arrays.npz")
    n_seeds = len({r["seed"] for r in rows})

    print(write_summary(rows))
    fig_test_error(rows)
    fig_training(arrays, n_seeds)
    fig_margins(arrays, n_seeds)
    fig_model_size(rows)
    print("\nSaved summary.txt and 4 figures to results/")
