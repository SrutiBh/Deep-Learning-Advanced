import numpy as np

from conformal.scoring import residual_norm


def normalize_weights(w_cal, w_test_point):
    """pi(u^(i)) := w(u^(i)) / (sum_j w(u^(j)) + w(u^(i)))  for i in 1..K,
    plus the implicit test-point weight for the R^(0)=inf term."""
    denom = np.sum(w_cal) + w_test_point
    pi_cal = w_cal / denom
    pi_test = w_test_point / denom
    return pi_cal, pi_test


def weighted_quantile(scores, weights, delta):
    """Weighted empirical quantile: smallest C such that the cumulative
    weight of {scores <= C} reaches 1-delta. The test point's own
    (infinite) score/weight mass is included via `weights` summing to
    < 1 -- the remaining mass implicitly sits at +inf, matching the
    R^(K+1)=inf convention of split CP.
    """
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    cum_weights = np.cumsum(sorted_weights)
    idx = np.searchsorted(cum_weights, 1 - delta)
    if idx >= len(sorted_scores):
        return np.inf
    return sorted_scores[idx]


def calibrate(y_cal, y_hat_cal, w_cal, w_test_point, delta):
    residuals = residual_norm(y_cal, y_hat_cal)
    pi_cal, _ = normalize_weights(w_cal, w_test_point)
    return weighted_quantile(residuals, pi_cal, delta)


def predict_interval(y_hat_test, C):
    return y_hat_test - C, y_hat_test + C


def predict_ball(y_hat_test, C):
    return y_hat_test, C
