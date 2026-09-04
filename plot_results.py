"""Reads results/summary.json (from run_experiments.py) and produces:
  - coverage vs target bar chart per benchmark/method
  - interval width comparison bar chart per benchmark/method
  - shift condition: vanilla vs weighted coverage comparison
Saves PNGs to config.RESULTS_DIR.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import config


def load_results():
    path = os.path.join(config.RESULTS_DIR, "summary.json")
    with open(path) as f:
        return json.load(f)


def plot_coverage(results, target=1 - config.DELTA):
    no_shift = [r for r in results if r["condition"] == "no_shift"]
    benchmarks = sorted(set(r["benchmark"] for r in no_shift))
    methods = sorted(set(r["method"] for r in no_shift))

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(methods)
    x = np.arange(len(benchmarks))
    for i, m in enumerate(methods):
        vals = []
        for b in benchmarks:
            match = [r["coverage"] for r in no_shift if r["benchmark"] == b and r["method"] == m]
            vals.append(match[0] if match else np.nan)
        ax.bar(x + i * width, vals, width, label=m)

    ax.axhline(target, color="black", linestyle="--", linewidth=1, label=f"target {target:.2f}")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage by method and benchmark (no shift)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(config.RESULTS_DIR, "coverage_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_interval_width(results):
    no_shift = [r for r in results if r["condition"] == "no_shift"]
    benchmarks = sorted(set(r["benchmark"] for r in no_shift))
    methods = sorted(set(r["method"] for r in no_shift))

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / len(methods)
    x = np.arange(len(benchmarks))
    for i, m in enumerate(methods):
        vals = []
        for b in benchmarks:
            match = [r["avg_width"] for r in no_shift if r["benchmark"] == b and r["method"] == m]
            vals.append(match[0] if match else np.nan)
        ax.bar(x + i * width, vals, width, label=m)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Average interval width / ball radius")
    ax.set_title("Efficiency by method and benchmark (no shift)")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(config.RESULTS_DIR, "width_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_shift_robustness(results, target=1 - config.DELTA):
    shift = [r for r in results if r["condition"] == "shift"]
    benchmarks = sorted(set(r["benchmark"] for r in shift))
    methods = sorted(set(r["method"] for r in shift))

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.8 / len(methods)
    x = np.arange(len(benchmarks))
    for i, m in enumerate(methods):
        vals = []
        for b in benchmarks:
            match = [r["coverage"] for r in shift if r["benchmark"] == b and r["method"] == m]
            vals.append(match[0] if match else np.nan)
        ax.bar(x + i * width, vals, width, label=m)

    ax.axhline(target, color="black", linestyle="--", linewidth=1, label=f"target {target:.2f}")
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(benchmarks)
    ax.set_ylabel("Empirical coverage under covariate shift")
    ax.set_title("Vanilla CP vs Weighted CP under shift")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(config.RESULTS_DIR, "shift_robustness.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    results = load_results()
    paths = [plot_coverage(results), plot_interval_width(results), plot_shift_robustness(results)]
    for p in paths:
        print(f"Saved: {p}")
