# Where Label Noise Sits: experiment code

Code for the preprint *Where Label Noise Sits: Far-From-Boundary Flips Hurt Gradient Descent More Than Near-Boundary Flips* (George Kavroulakis, 2026).

The experiments ask whether the **location** of mislabeled training points matters when the **amount** of label noise is held fixed. They compare two ways of flipping exactly 20% of the training labels:

- **Uniform noise:** flipped points are chosen uniformly at random.
- **Boundary-concentrated noise:** flips are more likely close to the true decision boundary. The flip probability decays as `exp(-|distance| / scale)`, and a smaller `scale` concentrates the flips more tightly.

Models are trained on the noisy labels with full-batch gradient descent on the logistic loss, then tested on fresh data with clean labels.

## Setup

Requires Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Running the experiments

Run everything from the repository's top-level folder, because results are written to a relative `results/` directory.

```
python step5_experiment.py
python step5_plots.py
python step5b_scale_sweep.py
```

1. `step5_experiment.py` runs the main experiment and writes `results/results.csv` and `results/arrays.npz`.
2. `step5_plots.py` reads those files and writes `results/summary.txt` and four figures (`fig1_test_error.png` to `fig4_model_size.png`).
3. `step5b_scale_sweep.py` repeats the boundary-noise runs at tighter concentration scales (0.5 and 0.25) and writes `results/scale_sweep.csv` and `results/scale_sweep_summary.txt`. It needs `results/results.csv` from step 1 above.

The experiments use multiple CPU cores and train many models, so a full run can take a while.

## What each file does

| File | Role |
|---|---|
| `step1_data_generation.py` | Samples points from a standard Gaussian and labels them with a fixed linear boundary, giving known clean labels. Running it directly prints a quick sanity check. |
| `step2_noise_injection.py` | Uniform and boundary-concentrated label noise, calibrated to the same flip rate (including exact-count versions). |
| `step3_training.py` | Full-batch gradient descent on the logistic loss for a linear model and a one-hidden-layer ReLU network. |
| `step4_metrics.py` | Test error (overall, near the boundary, far from it), normalized margins, weight norm, and the step at which training error reaches zero. |
| `step5_experiment.py` | Runs all setups across 10 seeds and three label conditions (clean, uniform, boundary). |
| `step5_plots.py` | Summary statistics and figures from the saved results. |
| `step5b_scale_sweep.py` | Checks whether the result holds when boundary noise is more tightly concentrated. |

`step1` to `step4` are modules imported by the step 5 scripts.

## Experimental setups

All setups use seeds 0 to 9 and flip exactly 20% of training labels.

| Setup | Model | Training points | Dimensions | Width |
|---|---|---|---|---|
| `net` | One-hidden-layer network | 200 | 10 | 128, 512, 1024 |
| `linear` | Linear (more dimensions than points) | 200 | 400, 800 | n/a |
| `linear_under` | Linear (more points than dimensions) | 2000 | 10 | n/a |

## Citation

If you use this code, please cite the preprint: https://doi.org/10.5281/zenodo.23873121

## Acknowledgements

Code written with assistance from Claude (Anthropic).
