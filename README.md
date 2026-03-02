# Triangular Panel Additive Model

This repo simulates and fits an additive model on a triangular panel where user trajectories have varying lengths. It compares the alternating least-squares estimator against a naive exposure-wise mean and includes optional visualizations of the true mean functions.

## Contents
- `main.py` — simulation (Option A), estimator, naive benchmark, bootstrap, and optional plotting helpers.

## Quick start
```bash
# (optional) create/activate an env, then install deps
pip install numpy matplotlib

python main.py
```

The script will:
- run two experiments: (1) decaying innovation variance (`sigma_decay=0.05`, plots on), (2) flat variance (`sigma_decay=0.0`, plots off by default),
- fit the additive model and the naive benchmark,
- bootstrap `b(j)` for CIs,
- print a table with true vs. estimated `b(j)` and pointwise MSEs,
- report overall MSE for both estimators.

To customize, edit the two `run_experiment(...)` calls in `main()` (e.g., change `sigma_decay`, `do_plots`, or sample size `n`).

## Visualizations
Plotting can be toggled per experiment via `do_plots` / `do_plots_3d` in `run_experiment`.
- `plot_true_curves`: true `a(t)` and `b(j)` lines.
- `plot_true_mean_surface`: 2D heatmap of the triangular mean surface.
- `plot_true_mean_surface_3d`: 3D surface plot.

## Notes
- Identifiability uses `b(1)=0`; estimates and truths follow the same convention.
- The user-cluster bootstrap uses Poisson(1) weights for speed and asymptotic equivalence to resampling users.
