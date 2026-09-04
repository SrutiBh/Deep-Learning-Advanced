import numpy as np
from sklearn.linear_model import LogisticRegression


def estimate_density_ratio(x_source, x_target):
    """Estimate w(x) = p_target(x) / p_source(x) via a probabilistic
    domain classifier (source=0, target=1). For a well-calibrated
    classifier, p(target|x)/p(source|x) * (n_source/n_target) is
    proportional to the true density ratio.

    Returns weights for the source-domain points (used as the
    calibration-set weights w(u^(i)) in Weighted CP).
    """
    x_all = np.concatenate([x_source, x_target], axis=0)
    labels = np.concatenate([
        np.zeros(len(x_source)), np.ones(len(x_target))
    ])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_all, labels)

    p_target_given_x = clf.predict_proba(x_source)[:, 1]
    p_target_given_x = np.clip(p_target_given_x, 1e-3, 1 - 1e-3)
    p_source_given_x = 1 - p_target_given_x

    n_source, n_target = len(x_source), len(x_target)
    w = (p_target_given_x / p_source_given_x) * (n_source / n_target)
    return w, clf


def test_point_weight(x_test_point, x_source, x_target, clf):
    """Density-ratio weight for a single held-out test point, using the
    already-fit domain classifier."""
    p_target_given_x = clf.predict_proba(x_test_point.reshape(1, -1))[:, 1]
    p_target_given_x = np.clip(p_target_given_x, 1e-3, 1 - 1e-3)
    p_source_given_x = 1 - p_target_given_x
    n_source, n_target = len(x_source), len(x_target)
    return float((p_target_given_x / p_source_given_x) * (n_source / n_target))
