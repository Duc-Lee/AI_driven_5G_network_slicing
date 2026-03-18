import numpy as np

class DemandBasedAllocation:
    def __init__(self, n_slices=3):
        self.n_slices = n_slices
        
    def select_action(self, state):
        # state[0:3] is traffic demand
        traffic = state[:self.n_slices]
        total_traffic = np.sum(traffic)
        if total_traffic == 0:
            return np.array([1.0/self.n_slices] * self.n_slices)
        # Proportional to demand
        ratios = traffic / total_traffic
        # Clamp to ensure minimum allocation
        ratios = np.maximum(ratios, 0.05)
        ratios = ratios / np.sum(ratios)
        return ratios
