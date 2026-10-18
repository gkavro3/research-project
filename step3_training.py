"""
Step 3: Train models with gradient descent under each noise condition.

For each label condition (clean, uniform noise, boundary-concentrated noise)
this trains two models with full-batch gradient descent on the logistic loss:

- Linear model: f(x) = w . x + b  (logistic regression)
- Shallow net:  f(x) = a . relu(W x + c) / sqrt(m)  (one hidden layer, width m)

Models are trained on the *noisy* labels and tested on fresh *clean* labels,
so test error measures how well the model recovered the true boundary.
Training curves (loss, train error, test error, weight norm) are recorded
for Step 4 (metrics) and Step 5 (plots).
"""

import numpy as np

from step1_data_generation import make_boundary, generate_clean_data, signed_distance_to_boundary
from step2_noise_injection import (
    inject_uniform_noise,
    inject_boundary_noise,
    inject_uniform_noise_exact,
    inject_boundary_noise_exact,
)


def logistic_loss_and_grad(margins: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean logistic loss log(1 + exp(-margin)) and its derivative w.r.t.
    each margin, where margin = y * f(x). Written with logaddexp so large
    margins don't overflow."""
    loss = np.mean(np.logaddexp(0.0, -margins))
    dloss = -np.exp(-np.logaddexp(0.0, margins))  # = -sigmoid(-margin)
    return loss, dloss


def error_rate(scores: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sign(scores) != y))


def train_linear(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    lr: float,
    steps: int,
    log_every: int,
) -> dict:
    """Logistic regression trained by full-batch gradient descent from zero."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    curves = {"step": [], "loss": [], "train_err": [], "test_err": [], "norm": []}

    for t in range(steps + 1):
        scores = X @ w + b
        loss, dloss = logistic_loss_and_grad(y * scores)

        if t % log_every == 0:
            curves["step"].append(t)
            curves["loss"].append(loss)
            curves["train_err"].append(error_rate(scores, y))
            curves["test_err"].append(error_rate(X_test @ w + b, y_test))
            curves["norm"].append(float(np.linalg.norm(w)))

        g = dloss * y / n  # d(loss)/d(score) for each point
        w -= lr * (X.T @ g)
        b -= lr * np.sum(g)

    return {"w": w, "b": b, "curves": {k: np.array(v) for k, v in curves.items()}}


def net_forward(X: np.ndarray, W: np.ndarray, c: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hidden = np.maximum(X @ W.T + c, 0.0)
    scores = hidden @ a / np.sqrt(len(a))
    return scores, hidden


def train_net(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    width: int,
    lr: float,
    steps: int,
    log_every: int,
    seed: int | None = None,
) -> dict:
    """One-hidden-layer ReLU net trained by full-batch gradient descent.
    Both layers are trained."""
    n, d = X.shape
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(width, d)) / np.sqrt(d)
    c = np.zeros(width)
    a = rng.normal(size=width)
    curves = {"step": [], "loss": [], "train_err": [], "test_err": [], "norm": []}

    for t in range(steps + 1):
        scores, hidden = net_forward(X, W, c, a)
        loss, dloss = logistic_loss_and_grad(y * scores)

        if t % log_every == 0:
            test_scores, _ = net_forward(X_test, W, c, a)
            curves["step"].append(t)
            curves["loss"].append(loss)
            curves["train_err"].append(error_rate(scores, y))
            curves["test_err"].append(error_rate(test_scores, y_test))
            curves["norm"].append(float(np.sqrt(np.sum(W**2) + np.sum(c**2) + np.sum(a**2))))

        g = dloss * y / n
        grad_a = hidden.T @ g / np.sqrt(width)
        back = np.outer(g, a / np.sqrt(width)) * (hidden > 0)
        grad_W = back.T @ X
        grad_c = back.sum(axis=0)

        a -= lr * grad_a
        W -= lr * grad_W
        c -= lr * grad_c

    return {"W": W, "c": c, "a": a, "curves": {k: np.array(v) for k, v in curves.items()}}


def make_conditions(n: int, d: int, p: float, scale: float, seed: int, exact: bool = False) -> dict:
    """Build the train set (three label versions) and a clean test set.
    Every random draw gets its own independent seed.

    exact=True flips exactly round(p * n) labels in both noise conditions
    instead of flipping each label independently."""
    s_boundary, s_train, s_test, s_uniform, s_bnoise = np.random.SeedSequence(seed).generate_state(5)

    w_true, b_true = make_boundary(d, seed=s_boundary)
    X, y_clean = generate_clean_data(n, d, w_true, b_true, seed=s_train)
    X_test, y_test = generate_clean_data(20000, d, w_true, b_true, seed=s_test)
    dist = signed_distance_to_boundary(X, w_true, b_true)

    if exact:
        y_uniform = inject_uniform_noise_exact(y_clean, p, seed=s_uniform)
        y_boundary = inject_boundary_noise_exact(y_clean, dist, p, scale, seed=s_bnoise)
        p_max = float("nan")
    else:
        y_uniform = inject_uniform_noise(y_clean, p, seed=s_uniform)
        y_boundary, p_max = inject_boundary_noise(y_clean, dist, p, scale, seed=s_bnoise)

    return {
        "X": X,
        "X_test": X_test,
        "y_test": y_test,
        "w_true": w_true,
        "b_true": b_true,
        "dist": dist,
        "p_max": p_max,
        "labels": {"clean": y_clean, "uniform": y_uniform, "boundary": y_boundary},
    }


if __name__ == "__main__":
    d = 10
    n = 2000
    seed = 0
    target_p = 0.2
    scale = 1.0
    width = 512

    data = make_conditions(n, d, target_p, scale, seed)
    X, X_test, y_test = data["X"], data["X_test"], data["y_test"]

    print(f"n={n}, d={d}, target flip rate={target_p}, scale={scale}, p_max={data['p_max']:.4f}")
    for name, y in data["labels"].items():
        print(f"  {name:9s} actual flip rate: {np.mean(y != data['labels']['clean']):.4f}")

    saved = {}
    for model in ("linear", "net"):
        print(f"\n{model}")
        print(f"  {'condition':9s} {'loss':>8s} {'train_err':>10s} {'test_err':>9s} {'norm':>8s}")
        for name, y in data["labels"].items():
            if model == "linear":
                out = train_linear(X, y, X_test, y_test, lr=0.5, steps=5000, log_every=50)
            else:
                out = train_net(X, y, X_test, y_test, width=width, lr=0.5, steps=5000, log_every=50, seed=seed)
            cv = out["curves"]
            print(f"  {name:9s} {cv['loss'][-1]:8.4f} {cv['train_err'][-1]:10.4f} "
                  f"{cv['test_err'][-1]:9.4f} {cv['norm'][-1]:8.2f}")
            for k, v in cv.items():
                saved[f"{model}_{name}_{k}"] = v

    np.savez("step3_curves.npz", **saved)
    print("\nSaved training curves to step3_curves.npz")
