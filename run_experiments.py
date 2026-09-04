"""Loads trained backbones, runs calibration through all four conformal
methods, evaluates on the appropriate test set(s), and writes a
comparative JSON/CSV summary table to config.RESULTS_DIR.

Usage: python run_experiments.py
Assumes train_backbones.py has already been run (checkpoints exist under
config.CHECKPOINT_DIR/<tag>/).
"""
import json
import os

import numpy as np
import pandas as pd
import torch

import config
from models.mlp import MLP
from models.quantile_net import QuantileNet
from models.variance_net import VarianceNet
from conformal import vanilla, cqr, locally_adaptive, weighted
from utils import metrics
from utils.density_ratio import estimate_density_ratio, test_point_weight


def load_backbones(tag, in_dim, out_dim):
    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, tag)
    mlp_model = MLP(in_dim, out_dim=out_dim)
    mlp_model.load_state_dict(torch.load(os.path.join(ckpt_dir, "mlp.pt")))
    mlp_model.eval()

    qnet = QuantileNet(in_dim, out_dim=out_dim)
    qnet.load_state_dict(torch.load(os.path.join(ckpt_dir, "quantile_net.pt")))
    qnet.eval()

    vnet = VarianceNet(in_dim, out_dim=1)  # always scalar sigma -- see train_backbones.py
    vnet.load_state_dict(torch.load(os.path.join(ckpt_dir, "variance_net.pt")))
    vnet.eval()

    return mlp_model, qnet, vnet


def _predict(model, x):
    with torch.no_grad():
        return model(torch.as_tensor(x)).numpy()


def _predict_quantiles(qnet, x):
    with torch.no_grad():
        ql, qu = qnet(torch.as_tensor(x))
        return ql.numpy(), qu.numpy()


def _predict_sigma(vnet, x):
    with torch.no_grad():
        return vnet(torch.as_tensor(x)).numpy().ravel()


def evaluate_no_shift(tag, mlp_model, qnet, vnet, x_cal, y_cal, x_test, y_test, delta, is_2d):
    """Vanilla CP, CQR, Locally Adaptive CP evaluated with matched
    calibration/test distributions (no covariate shift)."""
    y_hat_cal = _predict(mlp_model, x_cal)
    y_hat_test = _predict(mlp_model, x_test)
    ql_cal, qu_cal = _predict_quantiles(qnet, x_cal)
    ql_test, qu_test = _predict_quantiles(qnet, x_test)
    sigma_cal = _predict_sigma(vnet, x_cal)
    sigma_test = _predict_sigma(vnet, x_test)

    results = []

    # Vanilla
    C = vanilla.calibrate(y_cal, y_hat_cal, delta)
    if is_2d:
        cov = metrics.empirical_coverage_ball(y_test, y_hat_test, C)
        width = metrics.average_ball_radius(C)
    else:
        lo, hi = vanilla.predict_interval(y_hat_test.ravel(), C)
        cov = metrics.empirical_coverage(y_test.ravel(), lo, hi)
        width = metrics.average_interval_width(lo, hi)
    results.append({"method": "vanilla", "coverage": cov, "avg_width": width,
                     "coverage_deficit": (1 - delta) - cov, "C": float(C)})

    # CQR
    C = cqr.calibrate(y_cal, ql_cal, qu_cal, delta)
    if is_2d:
        lo, hi = cqr.predict_interval(ql_test, qu_test, C)
        dist = np.maximum(0, np.max(np.maximum(lo - y_test, y_test - hi), axis=1))
        cov = float(np.mean(dist == 0))
        width = float(np.mean(hi - lo))
    else:
        lo, hi = cqr.predict_interval(ql_test.ravel(), qu_test.ravel(), C)
        cov = metrics.empirical_coverage(y_test.ravel(), lo, hi)
        width = metrics.average_interval_width(lo, hi)
    results.append({"method": "cqr", "coverage": cov, "avg_width": width,
                     "coverage_deficit": (1 - delta) - cov, "C": float(C)})

    # Locally Adaptive
    C = locally_adaptive.calibrate(y_cal, y_hat_cal, sigma_cal, delta)
    if is_2d:
        radius = sigma_test * C  # shape (n,) -- do not reshape to (n,1), would broadcast wrong
        cov = metrics.empirical_coverage_ball(y_test, y_hat_test, radius)
        width = metrics.average_ball_radius(radius)
    else:
        lo, hi = locally_adaptive.predict_interval(y_hat_test.ravel(), sigma_test, C)
        cov = metrics.empirical_coverage(y_test.ravel(), lo, hi)
        width = metrics.average_interval_width(lo, hi)
    results.append({"method": "locally_adaptive", "coverage": cov, "avg_width": width,
                     "coverage_deficit": (1 - delta) - cov, "C": float(C)})

    for r in results:
        r["benchmark"] = tag
        r["condition"] = "no_shift"
    return results


def evaluate_weighted_shift(tag, mlp_model, x_cal, y_cal, x_test, y_test, delta):
    """Weighted CP compared against Vanilla CP under covariate shift."""
    y_hat_cal = _predict(mlp_model, x_cal)
    y_hat_test = _predict(mlp_model, x_test)

    # Vanilla CP under shift (expected to under-cover)
    C_vanilla = vanilla.calibrate(y_cal, y_hat_cal, delta)
    lo, hi = vanilla.predict_interval(y_hat_test.ravel(), C_vanilla)
    cov_vanilla = metrics.empirical_coverage(y_test.ravel(), lo, hi)

    # Weighted CP under shift
    w_cal, clf = estimate_density_ratio(x_cal, x_test)
    results = []
    for i in range(len(x_test)):
        w_test_i = test_point_weight(x_test[i], x_cal, x_test, clf)
        C = weighted.calibrate(y_cal, y_hat_cal, w_cal, w_test_i, delta)
        lo_i, hi_i = weighted.predict_interval(y_hat_test[i].ravel(), C)
        inside = (y_test[i] >= lo_i) & (y_test[i] <= hi_i)
        results.append({"inside": bool(np.all(inside)), "width": float(np.mean(hi_i - lo_i))})

    cov_weighted = float(np.mean([r["inside"] for r in results]))
    width_weighted = float(np.mean([r["width"] for r in results]))

    return [
        {"method": "vanilla", "benchmark": tag, "condition": "shift",
         "coverage": cov_vanilla, "avg_width": metrics.average_interval_width(lo, hi),
         "coverage_deficit": (1 - delta) - cov_vanilla, "C": float(C_vanilla)},
        {"method": "weighted", "benchmark": tag, "condition": "shift",
         "coverage": cov_weighted, "avg_width": width_weighted,
         "coverage_deficit": (1 - delta) - cov_weighted, "C": None},
    ]


def run_unicycle():
    from data.unicycle import load_unicycle_splits
    splits = load_unicycle_splits()
    mlp_model, qnet, vnet = load_backbones("unicycle", in_dim=4, out_dim=2)
    x_cal, y_cal = splits["cal"]
    x_test, y_test = splits["test"]
    return evaluate_no_shift("unicycle", mlp_model, qnet, vnet, x_cal, y_cal,
                              x_test, y_test, config.DELTA_UNICYCLE, is_2d=True)


def run_sin_toy():
    from data.sin_toy import load_sin_toy_splits, load_sin_toy_unshifted_splits
    mlp_model, qnet, vnet = load_backbones("sin_toy", in_dim=1, out_dim=1)

    # no-shift control condition
    unshifted = load_sin_toy_unshifted_splits()
    x_cal, y_cal = unshifted["cal"]
    x_test, y_test = unshifted["test"]
    results = evaluate_no_shift("sin_toy", mlp_model, qnet, vnet, x_cal, y_cal,
                                 x_test, y_test, config.DELTA, is_2d=False)

    # covariate shift condition (train source, cal/test shifted target)
    shifted = load_sin_toy_splits()
    x_cal_s, y_cal_s = shifted["cal"]
    x_test_s, y_test_s = shifted["test"]
    results += evaluate_weighted_shift("sin_toy", mlp_model, x_cal_s, y_cal_s,
                                        x_test_s, y_test_s, config.DELTA)
    return results


def run_parkinsons():
    from data.parkinsons import load_parkinsons_splits
    splits = load_parkinsons_splits()
    in_dim = len(splits["feature_cols"])
    mlp_model, qnet, vnet = load_backbones("parkinsons", in_dim=in_dim, out_dim=1)

    x_cal, y_cal = splits["cal_noshift"]
    x_test, y_test = splits["test_noshift"]
    results = evaluate_no_shift("parkinsons", mlp_model, qnet, vnet, x_cal, y_cal,
                                 x_test, y_test, config.DELTA, is_2d=False)

    x_cal_s, y_cal_s = splits["cal_shift"]
    x_test_s, y_test_s = splits["test_shift"]
    results += evaluate_weighted_shift("parkinsons", mlp_model, x_cal_s, y_cal_s,
                                        x_test_s, y_test_s, config.DELTA)
    return results


if __name__ == "__main__":
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    all_results = []
    all_results += run_unicycle()
    all_results += run_sin_toy()

    if os.path.exists(config.PARKINSONS_CSV_PATH):
        all_results += run_parkinsons()
    else:
        print(f"Skipping Parkinson's — {config.PARKINSONS_CSV_PATH} not found.")

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(config.RESULTS_DIR, "summary.csv")
    json_path = os.path.join(config.RESULTS_DIR, "summary.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(df.to_string(index=False))
    print(f"\nSaved: {csv_path}\n       {json_path}")
