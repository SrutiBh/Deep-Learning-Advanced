import torch
import torch.nn as nn


class QuantileNet(nn.Module):
    """Two-head network outputting (q_{delta/2}(x), q_{1-delta/2}(x))."""

    def __init__(self, in_dim, out_dim=1, hidden_dim=64, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.lower_head = nn.Linear(hidden_dim, out_dim)
        self.upper_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.trunk(x)
        return self.lower_head(h), self.upper_head(h)


def pinball_loss(y_true, y_pred, q):
    """Pinball / quantile loss for a single quantile level q in (0, 1)."""
    diff = y_true - y_pred
    return torch.mean(torch.maximum(q * diff, (q - 1) * diff))


def quantile_net_loss(y_true, q_lower_pred, q_upper_pred, delta):
    """Combined pinball loss for the (delta/2, 1-delta/2) quantile pair."""
    lower_loss = pinball_loss(y_true, q_lower_pred, delta / 2)
    upper_loss = pinball_loss(y_true, q_upper_pred, 1 - delta / 2)
    return lower_loss + upper_loss
