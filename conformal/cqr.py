import numpy as np

from conformal.vanilla import split_conformal_quantile
from conformal.scoring import cqr_score


def calibrate(y_cal, q_lower_cal, q_upper_cal, delta):
    """R^(i) = max(q_{delta/2}(u) - y, y - q_{1-delta/2}(u)), maxed
    across output dims for multi-dim targets."""
    scores = cqr_score(y_cal, q_lower_cal, q_upper_cal)
    return split_conformal_quantile(scores, delta)


def predict_interval(q_lower_test, q_upper_test, C):
    return q_lower_test - C, q_upper_test + C
