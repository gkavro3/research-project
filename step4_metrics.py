"""
Step 4: Metrics for a trained model.

Given a model from Step 3, measures:

- Test error on clean labels, overall and split into the half of test points
  nearest the true boundary vs. the half farthest from it.
- Normalized margins of the training points: y * f(x) divided by the size of
  the weights, so that simply scaling the weights up doesn't count as a
  bigger margin. Positive = classified as its (noisy) training label.
- Weight norm, and for the linear model the cosine between the learned and
  true weight vectors (1.0 = same direction).
- The first logged step at which training error reached zero (interpolation).
"""

import numpy as np

from step1_data_generation import signed_distance_to_boundary
from step3_training import net_forward, error_rate


def model_scores(model: dict, X: np.ndarray) -> np.ndarray:
    if "w" in model:
        return X @ model["w"] + model["b"]
    scores, _ = net_forward(X, model["W"], model["c"], model["a"])
    return scores


def normalized_margins(model: dict, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y * f(x) / ||weights||^L, where L is how the output scales when all
    weights are multiplied by a constant: L=1 for the linear model, L=2 for
    the one-hidden-layer net."""
    margins = y * model_scores(model, X)
    if "w" in model:
        return margins / np.linalg.norm(model["w"])
    sq_norm = np.sum(model["W"] ** 2) + np.sum(model["c"] ** 2) + np.sum(model["a"] ** 2)
    return margins / sq_norm


def interpolation_step(curves: dict) -> int:
    """First logged step with zero training error, or -1 if never reached."""
    hit = np.flatnonzero(curves["train_err"] == 0)
    return int(curves["step"][hit[0]]) if len(hit) else -1


def compute_metrics(model: dict, data: dict, y_train: np.ndarray) -> tuple[dict, np.ndarray]:
    """Returns (summary numbers, per-training-point normalized margins)."""
    X, X_test, y_test = data["X"], data["X_test"], data["y_test"]

    test_scores = model_scores(model, X_test)
    test_dist = np.abs(signed_distance_to_boundary(X_test, data["w_true"], data["b_true"]))
    near = test_dist < np.median(test_dist)

    margins = normalized_margins(model, X, y_train)
    curves = model["curves"]

    metrics = {
        "flip_rate": float(np.mean(y_train != data["labels"]["clean"])),
        "train_err": error_rate(model_scores(model, X), y_train),
        "test_err": error_rate(test_scores, y_test),
        "test_err_near": error_rate(test_scores[near], y_test[near]),
        "test_err_far": error_rate(test_scores[~near], y_test[~near]),
        "min_margin": float(margins.min()),
        "norm": float(curves["norm"][-1]),
        "interp_step": interpolation_step(curves),
        "cosine": float("nan"),
    }
    if "w" in model:
        metrics["cosine"] = float(model["w"] @ data["w_true"] / np.linalg.norm(model["w"]))
    return metrics, margins
