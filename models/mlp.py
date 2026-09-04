import torch
import torch.nn as nn


class MLP(nn.Module):
    """Base mean-prediction network f_theta(x) -> y_hat, trained with MSE."""

    def __init__(self, in_dim, out_dim=1, hidden_dim=64, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
