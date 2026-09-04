# Conformal Prediction with Deep Learning Backbones

Implements and compares four conformal prediction methods — Vanilla
Split CP, CQR, Locally Adaptive CP, and Weighted CP — across three
benchmarks: a synthetic Unicycle reachability task (paper's Example 1),
a synthetic heteroskedastic/covariate-shift sin(x) toy task, and the
real-world UCI Parkinson's Telemonitoring dataset.

## Setup

```
pip install -r requirements.txt
```

Parkinson's Telemonitoring requires manually downloading the CSV
(network access isn't available for this in some sandboxed
environments) from:
https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

Save it as `data_files/parkinsons_updrs.csv`. If this file is absent,
`run_experiments.py` automatically skips the Parkinson's benchmark and
still runs Unicycle + sin_toy.

## Running

```
python train_backbones.py     # trains MLP, QuantileNet, VarianceNet per benchmark
python run_experiments.py     # calibrates all four CP methods, evaluates, writes results/
python plot_results.py        # generates comparison plots from results/summary.json
```

Outputs land in `results/`:
- `summary.csv` / `summary.json` — per-method, per-benchmark coverage, interval width, coverage deficit
- `coverage_comparison.png` — empirical coverage vs. target, by method and benchmark
- `width_comparison.png` — interval width / ball radius, by method and benchmark
- `shift_robustness.png` — Vanilla CP vs. Weighted CP coverage under covariate shift

## Project layout

```
config.py              # seeds, delta levels, split sizes, paths
data/
  unicycle.py           # paper's Example 1 unicycle simulation
  sin_toy.py             # sin(x) heteroskedastic + covariate-shift toy data
  parkinsons.py           # UCI Parkinson's loader, disjoint train/val/cal/test splits
models/
  mlp.py                 # point-prediction backbone (MSE loss)
  quantile_net.py         # two-head quantile regression net (pinball loss), for CQR
  variance_net.py           # residual-magnitude estimator, for Locally Adaptive CP
conformal/
  scoring.py              # shared nonconformity-score helpers (residual_norm, cqr_score)
  vanilla.py                # R = |y - y_hat| (or its Euclidean-norm generalization)
  cqr.py                    # R = max(q_lo - y, y - q_hi)
  locally_adaptive.py         # R = |y - y_hat| / sigma(x)
  weighted.py                  # density-ratio-weighted quantile under covariate shift
utils/
  density_ratio.py         # domain-classifier-based w(x) estimation for weighted.py
  metrics.py                 # coverage, interval width, stratified coverage, coverage deficit
train_backbones.py          # trains and checkpoints all three networks per benchmark
run_experiments.py            # runs all four CP methods, writes results/summary.{csv,json}
plot_results.py                 # generates the three comparison plots
```

## Data splitting

Each benchmark keeps train / validation / calibration / test strictly
disjoint: `train` fits the backbone networks, `val` is used for
early-stopping during training, `cal` (or `cal_noshift` / `cal_shift`
for Parkinson's) computes the conformal quantile, and `test` (or
`test_noshift` / `test_shift`) is held out purely for evaluation.
Mixing any of these would invalidate the finite-sample coverage
guarantee.

## Notes on method scope

- Unicycle's target is 2D `(px, py)`; all four methods score it via the
  Euclidean norm of the residual vector, and evaluation uses a
  prediction *ball* (center + radius) rather than a symmetric interval.
- Weighted CP's density ratio `w(x)` is estimated via a logistic
  regression domain classifier (source vs. target), following the
  common practical approximation when the true likelihood ratio is
  unknown.
