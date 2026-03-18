import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .slice import eMBBSlice, URLLCSlice, mMTCSlice
from .traffic_generator import TrafficGenerator

"""
High-Fidelity 5G Network Slicing Environment with PRBs and SINR.
"""
class NetworkSlicingEnv(gym.Env):
    def __init__(self):
        super(NetworkSlicingEnv, self).__init__()
        # 5G NR Parameters (100MHz BW, 30kHz SCS = 273 PRBs)
        self.total_prbs = 273
        self.slice_names = ["eMBB", "URLLC", "mMTC"]
        self.slices = {
            "eMBB": eMBBSlice(),
            "URLLC": URLLCSlice(),
            "mMTC": mMTCSlice()
        }
        self.traffic_gen = TrafficGenerator(self.slice_names)
        self.action_space = spaces.Box(low=0.01, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(9,), dtype=np.float32)

    def _get_obs(self):
        demands = np.array([self.slices[name].traffic_demand for name in self.slice_names], dtype=np.float32) / 500.0
        sinrs = np.array([getattr(self.slices[name], 'avg_sinr', 0) for name in self.slice_names], dtype=np.float32) / 20.0
        allocs = np.array([self.slices[name].current_prbs for name in self.slice_names], dtype=np.float32) / self.total_prbs
        return np.concatenate([demands, sinrs, allocs])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for s in self.slices.values():
            s.traffic_demand = 0
            s.latency = 0
            s.throughput = 0
        traffic_data = self.traffic_gen.generate_traffic()
        for name, data in traffic_data.items():
            self.slices[name].calculate_metrics(data)   
        return self._get_obs(), {}

    def step(self, action):
        # Normalize action to sum to 1.0
        ratios = np.abs(action) / (np.sum(np.abs(action)) + 1e-6)
        for i, name in enumerate(self.slice_names):
            prbs = max(self.slices[name].min_prbs, int(ratios[i] * self.total_prbs))
            self.slices[name].update_resource(prbs)
        # Get new traffic and calculate metrics
        traffic_data = self.traffic_gen.generate_traffic()
        total_throughput = 0
        total_latency = 0
        total_qos_violation = 0
        total_waste = 0
        for name, data in traffic_data.items():
            self.slices[name].calculate_metrics(data)
            total_throughput += self.slices[name].throughput
            total_latency += self.slices[name].latency
            total_qos_violation += self.slices[name].qos_violations
            if self.slices[name].total_capacity > self.slices[name].traffic_demand:
                waste_ratio = 1.0 - (self.slices[name].traffic_demand / (self.slices[name].total_capacity + 1e-6))
                waste = self.slices[name].current_prbs * waste_ratio
            else:
                waste = 0
            total_waste += waste
        # Balanced Reward Function (Scaled down for stability)
        reward = ((total_throughput * 0.5) - (total_latency * 0.1) - (total_qos_violation * 50.0) - (total_waste * 0.05)) / 100.0
        done = False 
        truncated = False
        info = {
            "throughput": total_throughput,
            "latency": total_latency,
            "violations": total_qos_violation,
            "waste": total_waste,
            "slice_latencies": {name: self.slices[name].latency for name in self.slice_names},
            "bw_alloc": {name: self.slices[name].current_prbs for name in self.slice_names}
        }
        return self._get_obs(), reward, done, truncated, info
