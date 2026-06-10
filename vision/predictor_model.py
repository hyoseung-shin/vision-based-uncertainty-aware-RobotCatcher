import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int = 6, hidden: int = 64, n_horizons: int = 36):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True)
        self.mu_head    = nn.Linear(hidden, n_horizons * 3)
        self.sigma_head = nn.Linear(hidden, n_horizons * 3)
        self.n_horizons = n_horizons

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        mu        = self.mu_head(last).view(-1, self.n_horizons, 3)
        log_sigma = self.sigma_head(last).view(-1, self.n_horizons, 3)
        log_sigma = torch.clamp(log_sigma, min=-6.9, max=0.0)
        return mu, log_sigma
