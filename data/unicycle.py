import numpy as np

import config


def _truncated_normal(mean, std, low, high, size, rng):
    x = rng.normal(mean, std, size=size)
    # simple resample-out-of-bounds truncation
    mask = (x < low) | (x > high)
    while mask.any():
        x[mask] = rng.normal(mean, std, size=mask.sum())
        mask = (x < low) | (x > high)
    return x


def _simulate(n, rng, horizon=config.UNICYCLE_HORIZON, dt=config.UNICYCLE_DT,
              omega=config.UNICYCLE_OMEGA, noise_std=config.UNICYCLE_NOISE_STD):
    """Iterates the true discrete-time unicycle dynamics for `horizon`
    steps (not a closed-form shortcut):
        theta_{t+1} = theta_t + omega * dt
        p_{t+1}^x   = p_t^x + v * cos(theta_t) * dt + w_t^x
        p_{t+1}^y   = p_t^y + v * sin(theta_t) * dt + w_t^y
    with w_t ~ N(0, noise_std^2) drawn fresh at every step.

    Input distribution D_phi_in matches the paper exactly:
        px0, py0 ~ Uniform([0, 1))
        theta0   ~ N(0, 0.1^2)
        v0       ~ TruncatedNormal(1, 0.1^2) on [0, 2]
    omega is a fixed constant (not part of u), since the paper doesn't
    include it in u := (px0, py0, theta0, v0) and never states its value.
    """
    x0 = rng.uniform(config.UNICYCLE_POS_LOW, config.UNICYCLE_POS_HIGH, size=n)
    y0 = rng.uniform(config.UNICYCLE_POS_LOW, config.UNICYCLE_POS_HIGH, size=n)
    theta0 = rng.normal(config.UNICYCLE_THETA_MEAN, config.UNICYCLE_THETA_STD, size=n)
    v = _truncated_normal(config.UNICYCLE_V_MEAN, config.UNICYCLE_V_STD,
                           *config.UNICYCLE_V_CLIP, n, rng)

    theta = theta0.copy()
    px, py = x0.copy(), y0.copy()
    for _ in range(horizon):
        px = px + v * np.cos(theta) * dt + rng.normal(0, noise_std, size=n)
        py = py + v * np.sin(theta) * dt + rng.normal(0, noise_std, size=n)
        theta = theta + omega * dt

    u = np.stack([x0, y0, theta0, v], axis=1)  # matches paper's u := (px0, py0, theta0, v0)
    y = np.stack([px, py], axis=1)
    return u.astype(np.float32), y.astype(np.float32)


def load_unicycle_splits(n_train=5000, n_val=1000, n_cal=2000, n_test=config.N_TEST, seed=config.SEED):
    rng = np.random.default_rng(seed)
    u_train, y_train = _simulate(n_train, rng)
    u_val, y_val = _simulate(n_val, rng)
    u_cal, y_cal = _simulate(n_cal, rng)
    u_test, y_test = _simulate(n_test, rng)
    return {
        "train": (u_train, y_train),
        "val": (u_val, y_val),
        "cal": (u_cal, y_cal),
        "test": (u_test, y_test),
    }
