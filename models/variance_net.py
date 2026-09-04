import torch
import torch.nn as nn


class VarianceNet(nn.Module):
    """Auxiliary heteroskedastic noise estimator sigma_phi(x).

    Predicts the expected residual magnitude |y - f_theta(x)|; softplus
    keeps the output strictly positive so it can safely divide residuals
    in Locally Adaptive CP.
    """

    def __init__(self, in_dim, out_dim=1, hidden_dim=64, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
        self.eps = 1e-3  # floor to avoid division by ~0

    def forward(self, x):
        return self.softplus(self.net(x)) + self.eps


def variance_net_loss(residual_true, sigma_pred):
    """Smooth L1 loss between the observed residual magnitude and the
    network's predicted magnitude."""
    return nn.functional.smooth_l1_loss(sigma_pred, residual_true)
