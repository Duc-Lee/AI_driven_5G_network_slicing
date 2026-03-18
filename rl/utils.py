from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import math
import torch
import torch.nn.functional as F

@dataclass
class FreqBounds:
    f_min: float
    f_max: float
    def to_tensor(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.f_min, device=device), torch.tensor(self.f_max, device=device)

def squash_tanh(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    # Map R -> (-1,1) via tanh, then to [low, high]
    s = torch.tanh(x)
    return (s + 1.0) * 0.5 * (high - low) + low

def unsquash_tanh(y: torch.Tensor, low: float, high: float) -> torch.Tensor:
    # Map [low, high] back to R approximately
    y = (y - low) / max(1e-12, (high - low)) # [0,1]
    y = y * 2.0 - 1.0 # [-1,1]
    return 0.5 * torch.log((1 + y + 1e-12) / (1 - y + 1e-12))

def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

class SquashedNormal:
    def __init__(self, mu: torch.Tensor, log_std: torch.Tensor, low: float, high: float):
        self.mu = mu
        self.log_std = log_std.clamp(-20, 2)
        self.std = self.log_std.exp()
        self.low = low
        self.high = high
        self.base = torch.distributions.Normal(self.mu, self.std)

    def rsample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.base.rsample() 
        # tanh then scale to [low, high]
        y = squash_tanh(z, self.low, self.high)
        # for u = tanh(z) in (-1,1), y = (u+1)/2*(high-low)+low
        # |dy/dz| = |(high-low)/2 * (1 - tanh(z)^2)|
        log_prob = self.base.log_prob(z) - torch.log(1 - torch.tanh(z) ** 2 + 1e-12)
        # add constant scale term (high-low)/2
        log_prob = log_prob - math.log((self.high - self.low) / 2.0 + 1e-12)
        log_prob = log_prob.sum(-1)
        return y, log_prob

    def mode(self) -> torch.Tensor:
        return squash_tanh(self.mu, self.low, self.high)

def quantile_huber_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """QR-DQN style quantile Huber loss.
    pred: [B, N] quantiles, target: [B, N] (stopgrad), taus: [N]
    """
    # [B, N_tgt, N_pred]
    delta = target.unsqueeze(2) - pred.unsqueeze(1)  
    abs_delta = torch.abs(delta)
    huber = torch.where(abs_delta <= 1.0, 0.5 * delta ** 2, (abs_delta - 0.5))
    # [1, N_tgt, 1]
    tau = taus.view(1, -1, 1)  
    weight = torch.abs((delta.detach() < 0).float() - tau)
    loss = (weight * huber).mean()
    return loss
