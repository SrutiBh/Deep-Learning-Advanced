import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config
from models.mlp import MLP
from models.quantile_net import QuantileNet, quantile_net_loss
from models.variance_net import VarianceNet, variance_net_loss


def _as_target(y):
    """Ensures targets are 2D (n, out_dim) — Unicycle's y is already
    (n, 2); 1D targets like sin_toy/Parkinson's get reshaped to (n, 1)."""
    y_t = torch.as_tensor(y)
    if y_t.ndim == 1:
        y_t = y_t.reshape(-1, 1)
    return y_t


def _out_dim(y):
    return y.shape[1] if y.ndim > 1 else 1


def _loader(x, y, batch_size, shuffle=True):
    ds = TensorDataset(torch.as_tensor(x), _as_target(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_mlp(x_train, y_train, x_val, y_val, in_dim, epochs=config.EPOCHS, lr=config.LR):
    model = MLP(in_dim, out_dim=_out_dim(y_train))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = _loader(x_train, y_train, config.BATCH_SIZE)
    x_val_t = torch.as_tensor(x_val)
    y_val_t = _as_target(y_val)

    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(x_val_t), y_val_t).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_quantile_net(x_train, y_train, x_val, y_val, in_dim, delta,
                        epochs=config.EPOCHS, lr=config.LR):
    model = QuantileNet(in_dim, out_dim=_out_dim(y_train))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = _loader(x_train, y_train, config.BATCH_SIZE)
    x_val_t = torch.as_tensor(x_val)
    y_val_t = _as_target(y_val)

    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            ql, qu = model(xb)
            loss = quantile_net_loss(yb, ql, qu, delta)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            ql, qu = model(x_val_t)
            val_loss = quantile_net_loss(y_val_t, ql, qu, delta).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def _residual_magnitude(y_target, y_hat):
    """Scalar per-sample residual magnitude: |y - y_hat| for 1D targets,
    ||y - y_hat||_2 for multi-dim targets (matches
    conformal.scoring.residual_norm's convention, computed in torch)."""
    diff = y_target - y_hat
    if diff.shape[1] == 1:
        return diff.abs()
    return diff.norm(dim=1, keepdim=True)


def train_variance_net(x_train, y_train, x_val, y_val, in_dim, mlp_model,
                        epochs=config.EPOCHS, lr=config.LR):
    """Trains sigma_phi(x) to predict the scalar residual magnitude
    (|y - f_theta(x)| for 1D targets, ||y - f_theta(x)||_2 for multi-dim
    targets like Unicycle) -- always out_dim=1, since Locally Adaptive CP
    divides the residual norm by a single scalar sigma per sample."""
    mlp_model.eval()
    with torch.no_grad():
        resid_train = _residual_magnitude(_as_target(y_train), mlp_model(torch.as_tensor(x_train)))
        resid_val = _residual_magnitude(_as_target(y_val), mlp_model(torch.as_tensor(x_val)))

    model = VarianceNet(in_dim, out_dim=1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(torch.as_tensor(x_train), resid_train)
    train_loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True)
    x_val_t = torch.as_tensor(x_val)

    best_val, best_state = float("inf"), None
    for epoch in range(epochs):
        model.train()
        for xb, rb in train_loader:
            opt.zero_grad()
            loss = variance_net_loss(rb, model(xb))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = variance_net_loss(resid_val, model(x_val_t)).item()
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_all_backbones(splits, in_dim, delta, tag):
    x_train, y_train = splits["train"]
    x_val, y_val = splits["val"]

    print(f"[{tag}] training MLP...")
    mlp_model = train_mlp(x_train, y_train, x_val, y_val, in_dim)

    print(f"[{tag}] training QuantileNet...")
    qnet = train_quantile_net(x_train, y_train, x_val, y_val, in_dim, delta)

    print(f"[{tag}] training VarianceNet...")
    vnet = train_variance_net(x_train, y_train, x_val, y_val, in_dim, mlp_model)

    ckpt_dir = os.path.join(config.CHECKPOINT_DIR, tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(mlp_model.state_dict(), os.path.join(ckpt_dir, "mlp.pt"))
    torch.save(qnet.state_dict(), os.path.join(ckpt_dir, "quantile_net.pt"))
    torch.save(vnet.state_dict(), os.path.join(ckpt_dir, "variance_net.pt"))

    return mlp_model, qnet, vnet


if __name__ == "__main__":
    from data.unicycle import load_unicycle_splits
    from data.sin_toy import load_sin_toy_splits

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    uni_splits = load_unicycle_splits()
    train_all_backbones(uni_splits, in_dim=4, delta=config.DELTA_UNICYCLE, tag="unicycle")

    toy_splits = load_sin_toy_splits()
    train_all_backbones(toy_splits, in_dim=1, delta=config.DELTA, tag="sin_toy")

    print("Parkinson's requires data_files/parkinsons_updrs.csv — run separately once downloaded.")
