import numpy as np


def residual_norm(y, y_hat):
    """R^(i) = ||y - y_hat||_2. Reduces to |y - y_hat| when y is 1D
    (sin_toy, Parkinson's); returns the Euclidean distance when y is 2D
    (Unicycle's (px, py))."""
    y = np.atleast_2d(y) if y.ndim == 1 and y_hat.ndim == 1 else y
    diff = y - y_hat
    if diff.ndim == 1:
        return np.abs(diff)
    return np.linalg.norm(diff, axis=1)


def cqr_score(y, q_lower, q_upper):
    """R^(i) = max(q_lower - y, y - q_upper), taken elementwise then
    maxed across output dimensions for multi-dim targets (Unicycle) so a
    single scalar score is produced per sample, same convention as
    residual_norm."""
    per_dim = np.maximum(q_lower - y, y - q_upper)
    if per_dim.ndim == 1:
        return per_dim
    return np.max(per_dim, axis=1)


def cqr_interval(y_hat_dummy_shape, q_lower, q_upper, C):
    """Expands both bounds by C in every output dimension."""
    return q_lower - C, q_upper + C
