import numpy as np


def empirical_coverage(y_true, lower, upper):
    inside = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(inside))


def average_interval_width(lower, upper):
    return float(np.mean(upper - lower))


def empirical_coverage_ball(y_true, center, radius):
    """For norm-scored multi-dim targets: coverage = fraction of points
    within a ball of given radius around the predicted center."""
    dist = np.linalg.norm(y_true - center, axis=1)
    return float(np.mean(dist <= radius))


def average_ball_radius(radius):
    radius = np.atleast_1d(radius)
    return float(np.mean(radius))


def stratified_conditional_coverage(x, y_true, lower, upper, n_bins=10):
    """Empirical coverage computed within each bin of a 1D covariate x.
    For multi-dim x, pass a single representative feature (e.g. the
    feature most linked to heteroskedasticity)."""
    x = np.asarray(x).ravel()
    bin_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    bin_edges[-1] += 1e-8  # include the max point in the last bin
    bin_idx = np.digitize(x, bin_edges[1:-1])

    results = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        cov = empirical_coverage(y_true[mask], lower[mask], upper[mask])
        width = average_interval_width(lower[mask], upper[mask])
        results.append({
            "bin": b,
            "x_range": (float(bin_edges[b]), float(bin_edges[b + 1])),
            "n": int(mask.sum()),
            "coverage": cov,
            "avg_width": width,
        })
    return results


def coverage_deficit(y_true, lower, upper, target_coverage):
    """How far below the target 1-delta the achieved coverage falls
    (positive = under-covering / guarantee violated)."""
    achieved = empirical_coverage(y_true, lower, upper)
    return float(target_coverage - achieved)


def summarize(name, y_true, lower, upper, target_coverage, x_for_strat=None):
    summary = {
        "method": name,
        "coverage": empirical_coverage(y_true, lower, upper),
        "avg_interval_width": average_interval_width(lower, upper),
        "coverage_deficit": coverage_deficit(y_true, lower, upper, target_coverage),
    }
    if x_for_strat is not None:
        summary["stratified_coverage"] = stratified_conditional_coverage(
            x_for_strat, y_true, lower, upper
        )
    return summary
