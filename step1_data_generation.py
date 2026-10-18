"""
Step 1: Data generation for the label-noise geometry experiment.

Samples x ~ N(0, I_d) and labels each point with a fixed, known linear
decision boundary sign(w . x + b). This gives ground-truth clean labels
and a known Bayes-optimal boundary, which Step 2 (noise injection) will
corrupt in two ways (uniform vs. boundary-concentrated) for comparison.
"""

import numpy as np


def make_boundary(d: int, seed: int | None = None) -> tuple[np.ndarray, float]:
    """Sample a random unit-norm weight vector w and bias b defining the
    ground-truth linear boundary sign(w . x + b)."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=d)
    w = w / np.linalg.norm(w)
    b = 0.0  # boundary through the origin, keeps class balance ~50/50
    return w, b


def generate_clean_data(
    n: int, d: int, w: np.ndarray, b: float, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Sample n points x ~ N(0, I_d) and label them with sign(w . x + b).

    Returns:
        X: (n, d) array of points
        y_clean: (n,) array of labels in {-1, +1}
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    scores = X @ w + b
    y_clean = np.sign(scores)
    y_clean[y_clean == 0] = 1  # break ties (near-zero prob with continuous data)
    return X, y_clean


def signed_distance_to_boundary(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Distance of each point to the boundary w . x + b = 0.

    Since ||w|| = 1, this is just w . x + b. Step 2's boundary-concentrated
    noise will use |this value| to decide flip probability.
    """
    return X @ w + b


if __name__ == "__main__":
    d = 10       # dimensionality — placeholder, revisit once Step 5 (widths) is scoped
    n = 2000     # sample size — placeholder, revisit once Step 5 (seeds) is scoped
    seed = 0

    w, b = make_boundary(d, seed=seed)
    X, y_clean = generate_clean_data(n, d, w, b, seed=seed)
    dist = signed_distance_to_boundary(X, w, b)

    print(f"X shape: {X.shape}, y_clean shape: {y_clean.shape}")
    print(f"Class balance: {np.mean(y_clean == 1):.3f} positive")
    print(f"Distance to boundary: min={dist.min():.3f}, max={dist.max():.3f}, "
          f"mean|dist|={np.mean(np.abs(dist)):.3f}")
