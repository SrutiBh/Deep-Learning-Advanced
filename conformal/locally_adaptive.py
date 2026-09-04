import numpy as np

from conformal.vanilla import split_conformal_quantile
from conformal.scoring import residual_norm


def calibrate(y_cal, y_hat_cal, sigma_cal, delta):
    """R^(i) = ||y - mu(u)|| / sigma(u). sigma_cal must be a scalar
    per-sample estimate (VarianceNet's output), broadcasting over the
    residual norm for multi-dim targets."""
    scores = residual_norm(y_cal, y_hat_cal) / sigma_cal
    return split_conformal_quantile(scores, delta)


def predict_interval(y_hat_test, sigma_test, C):
    return y_hat_test - sigma_test * C, y_hat_test + sigma_test * C


def predict_ball(y_hat_test, sigma_test, C):
    """Radius sigma(x) * C, for norm-scored multi-dim targets."""
    return y_hat_test, sigma_test * C
