"""
Step 2: Noise injection for the label-noise geometry experiment.

Two ways to corrupt y_clean into y_noisy, calibrated to the same average
flip rate so the comparison isolates *where* the noise sits, not *how much*:

- Uniform noise: every point flipped independently w.p. p.
- Boundary-concentrated noise: flip probability decays with distance to
  the boundary (near-boundary points flip often, far points rarely),
  calibrated empirically (Option A) so the average flip rate over the
  actual sample equals p exactly.
"""

import numpy as np

from step1_data_generation import make_boundary, generate_clean_data, signed_distance_to_boundary


def inject_uniform_noise(
    y_clean: np.ndarray, p: float, seed: int | None = None
) -> np.ndarray:
    """Flip each label independently with probability p."""
    rng = np.random.default_rng(seed)
    flips = rng.random(size=y_clean.shape) < p
    y_noisy = y_clean.copy()
    y_noisy[flips] *= -1
    return y_noisy


def inject_boundary_noise(
    y_clean: np.ndarray,
    dist: np.ndarray,
    p: float,
    scale: float,
    seed: int | None = None,
) -> np.ndarray:
    """Flip labels with probability that decays with |distance to boundary|,
    calibrated (empirically, on this actual sample) so the average flip
    rate equals p.

    flip_prob(dist) = p_max * exp(-|dist| / scale)

    Calibration: try p_max = 1, measure the average of exp(-|dist|/scale)
    over the real sample, then rescale so the average hits exactly p.
    Works because the average scales linearly with p_max.
    """
    raw_shape = np.exp(-np.abs(dist) / scale)  # p_max = 1 version
    avg_raw = np.mean(raw_shape)
    p_max = p / avg_raw

    if p_max > 1.0:
        raise ValueError(
            f"Calibration requires p_max={p_max:.3f} > 1, which isn't a valid "
            f"probability. Increase `scale` (spreads flips further from the "
            f"boundary, lowering the peak needed) or lower `p`."
        )

    flip_prob = p_max * raw_shape
    rng = np.random.default_rng(seed)
    flips = rng.random(size=y_clean.shape) < flip_prob
    y_noisy = y_clean.copy()
    y_noisy[flips] *= -1
    return y_noisy, p_max


def inject_uniform_noise_exact(
    y_clean: np.ndarray, p: float, seed: int | None = None
) -> np.ndarray:
    """Flip exactly round(p * n) labels, chosen uniformly at random."""
    rng = np.random.default_rng(seed)
    n = len(y_clean)
    flips = rng.choice(n, size=round(p * n), replace=False)
    y_noisy = y_clean.copy()
    y_noisy[flips] *= -1
    return y_noisy


def inject_boundary_noise_exact(
    y_clean: np.ndarray,
    dist: np.ndarray,
    p: float,
    scale: float,
    seed: int | None = None,
) -> np.ndarray:
    """Flip exactly round(p * n) labels, chosen at random with weight
    exp(-|dist| / scale), so near-boundary points are picked more often.

    Same decay shape as inject_boundary_noise, but the flip *count* is fixed
    instead of random, so both noise conditions flip the same number of
    labels in every run.
    """
    rng = np.random.default_rng(seed)
    n = len(y_clean)
    weights = np.exp(-np.abs(dist) / scale)
    flips = rng.choice(n, size=round(p * n), replace=False, p=weights / weights.sum())
    y_noisy = y_clean.copy()
    y_noisy[flips] *= -1
    return y_noisy


if __name__ == "__main__":
    d = 10
    n = 2000
    seed = 0
    target_p = 0.2
    scale = 1.0  # placeholder — controls how tightly noise concentrates near boundary

    w, b = make_boundary(d, seed=seed)
    X, y_clean = generate_clean_data(n, d, w, b, seed=seed)
    dist = signed_distance_to_boundary(X, w, b)

    y_uniform = inject_uniform_noise(y_clean, target_p, seed=seed)
    y_boundary, p_max = inject_boundary_noise(y_clean, dist, target_p, scale, seed=seed)

    actual_uniform_rate = np.mean(y_uniform != y_clean)
    actual_boundary_rate = np.mean(y_boundary != y_clean)

    print(f"Target flip rate: {target_p}")
    print(f"Uniform noise actual rate: {actual_uniform_rate:.4f}")
    print(f"Boundary noise actual rate: {actual_boundary_rate:.4f} (p_max={p_max:.4f})")

    # Sanity check: boundary noise should flip near-boundary points more often
    near = np.abs(dist) < np.median(np.abs(dist))
    far = ~near
    print(f"Boundary noise flip rate, near-boundary half: {np.mean(y_boundary[near] != y_clean[near]):.4f}")
    print(f"Boundary noise flip rate, far-from-boundary half: {np.mean(y_boundary[far] != y_clean[far]):.4f}")
