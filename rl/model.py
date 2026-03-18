import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import SquashedNormal

class ContinuousActor(nn.Module):
    """
    actor network
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256, low=0.01, high=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.low = low
        self.high = high

    def forward(self, state):
        h = self.net(state)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        dist = SquashedNormal(mu, log_std, self.low, self.high)
        # sample y and log_prob
        action, log_prob = dist.rsample()
        return action, log_prob

    def deterministic(self, state):
        mu, log_std = self.forward(state)
        dist = SquashedNormal(mu, log_std, self.low, self.high)
        return dist.mode()

class QuantileCritic(nn.Module):
    """
    critic network
    """
    def __init__(self, obs_dim, action_dim, n_quantiles=32, hidden_dim=256):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_quantiles)
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_quantiles)
        )
        
    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)
