import numpy as np

import config


def _sample(n, mean, std, rng):
    x = rng.normal(mean, std, size=n)
    noise_scale = 0.1 + 0.5 * np.abs(x)
    y = np.sin(x) + rng.normal(0, noise_scale)
    return x.astype(np.float32).reshape(-1, 1), y.astype(np.float32)


def load_sin_toy_splits(n_train=5000, n_val=1000, n_cal=2000, n_test=config.N_TEST, seed=config.SEED):
    """Train drawn from N(-1,1); val/cal/test drawn from the shifted N(1,1)
    distribution, so calibration and test share the shift (matches the
    train-time-only shift setup) while train stays at the source domain.
    """
    rng = np.random.default_rng(seed)
    x_train, y_train = _sample(n_train, config.TOY_TRAIN_MEAN, config.TOY_TRAIN_STD, rng)
    x_val, y_val = _sample(n_val, config.TOY_SHIFT_MEAN, config.TOY_SHIFT_STD, rng)
    x_cal, y_cal = _sample(n_cal, config.TOY_SHIFT_MEAN, config.TOY_SHIFT_STD, rng)
    x_test, y_test = _sample(n_test, config.TOY_SHIFT_MEAN, config.TOY_SHIFT_STD, rng)
    return {
        "train": (x_train, y_train),
        "val": (x_val, y_val),
        "cal": (x_cal, y_cal),
        "test": (x_test, y_test),
        "train_source_mean_std": (config.TOY_TRAIN_MEAN, config.TOY_TRAIN_STD),
        "shift_mean_std": (config.TOY_SHIFT_MEAN, config.TOY_SHIFT_STD),
    }


def load_sin_toy_unshifted_splits(n_train=5000, n_val=1000, n_cal=2000, n_test=config.N_TEST, seed=config.SEED):
    """All splits drawn from the same N(-1,1) source distribution — the
    no-shift control condition, used to confirm Vanilla CP achieves
    correct coverage absent covariate shift."""
    rng = np.random.default_rng(seed)
    splits = {}
    for name, n in (("train", n_train), ("val", n_val), ("cal", n_cal), ("test", n_test)):
        splits[name] = _sample(n, config.TOY_TRAIN_MEAN, config.TOY_TRAIN_STD, rng)
    return splits
