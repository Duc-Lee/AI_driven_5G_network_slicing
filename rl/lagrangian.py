import torch
from dataclasses import dataclass

@dataclass
class ConstraintSpec:
    name: str
    target: float  
    kp: float = 0.01
    ki: float = 0.001
    kd: float = 0.0
    clamp_max: float = 5.0

class PIDLagrangian:
    """
    adjust lambda to satisfy constraints
    """
    def __init__(self, constraints: list[ConstraintSpec]):
        self.constraints = constraints
        self.lmbda = {c.name: torch.tensor(0.0) for c in constraints}
        self.err_int = {c.name: 0.0 for c in constraints}
        self.err_prev = {c.name: 0.0 for c in constraints}

    def compute_penalty(self, cost_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        r_eff = r - sum(lambda * penalty)
        penalty = max(0, cost - target)
        """
        first_tensor = next(iter(cost_dict.values()))
        penalty_total = torch.zeros_like(first_tensor)
        for spec in self.constraints:
            cost = cost_dict[spec.name]
            # Violation exists if cost > target
            violation = (cost - spec.target).clamp(min=0.0)
            penalty_total = penalty_total + self.lmbda[spec.name].to(cost.device) * violation
        return penalty_total

    def update(self, cost_dict: dict[str, torch.Tensor]) -> dict[str, float]:
        stats = {}
        for spec in self.constraints:
            name = spec.name
            current_cost = cost_dict[name].mean().item()
            error = max(0.0, current_cost - spec.target)
            # PID Update
            self.err_int[name] += error
            diff = error - self.err_prev[name]
            self.err_prev[name] = error
            delta = spec.kp * error + spec.ki * self.err_int[name] + spec.kd * diff
            new_val = max(0.0, min(spec.clamp_max, float(self.lmbda[name].item()) + delta))
            self.lmbda[name] = torch.tensor(new_val)
            stats[f"lambda_{name}"] = new_val
            stats[f"cost_{name}"] = current_cost   
        return stats
