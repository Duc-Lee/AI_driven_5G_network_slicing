import numpy as np

class StaticAllocation:
    def __init__(self, n_slices=3):
        self.n_slices = n_slices
       
    def select_action(self, state):
        # Equal split: 1/3 for each
        return np.array([1.0/self.n_slices] * self.n_slices)
