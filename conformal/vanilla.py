import numpy as np

from conformal.scoring import residual_norm


def split_conformal_quantile(scores, delta):
    """Vanilla split-CP quantile: Quantile_{1-delta}(R^(1),...,R^(K), inf).

    Uses the finite-sample-correct rank (ceil((K+1)(1-delta))) rather than
    a plain np.quantile call, which is what gives the 1-delta coverage
    guarantee at finite K.
    """
    scores = np.sort(np.asarray(scores))
    k = len(scores)
    rank = int(np.ceil((k + 1) * (1 - delta)))
    if rank >= k:
        return np.inf
    return scores[rank - 1]


def calibrate(y_cal, y_hat_cal, delta):
    """R_i = |y_i - y_hat_i| (or its Euclidean-norm generalization for
    multi-dim targets, via conformal.scoring.residual_norm)."""
    scores = residual_norm(y_cal, y_hat_cal)
    return split_conformal_quantile(scores, delta)


def predict_interval(y_hat_test, C):
    """For 1D targets: symmetric interval y_hat +/- C."""
    return y_hat_test - C, y_hat_test + C


def predict_ball(y_hat_test, C):
    """Returns center and radius of the prediction ball for norm-scored
    multi-dim targets (e.g. Unicycle's (px, py))."""
    return y_hat_test, C
