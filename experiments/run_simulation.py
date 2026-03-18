import os
import numpy as np
import pandas as pd
from env.network_env import NetworkSlicingEnv
from rl.agent import SlicingSAC, ReplayBuffer
from baseline.static_allocation import StaticAllocation
from baseline.demand_based import DemandBasedAllocation
from rl.train_sac import train

def run_comparison():
    env = NetworkSlicingEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Train SAC 
    agent, _, _ = train(num_episodes=300, max_steps=500, save_curves=False)
    # Compare Algorithms
    algorithms = {
        "Static": StaticAllocation(),
        "Demand-Based": DemandBasedAllocation(),
        "SAC": agent
    }
    results = []
    for name, algo in algorithms.items():
        print(f"Running Evaluation for {name}...")
        state, _ = env.reset()
        ep_reward = 0
        ep_latency = 0
        ep_throughput = 0
        ep_violations = 0
        for step in range(500):
            if name == "SAC":
                action = algo.select_action(state, deterministic=True)
            else:
                action = algo.select_action(state) 
            next_state, reward, done, _, info = env.step(action)
            ep_reward += reward
            ep_latency += info["latency"]
            ep_throughput += info["throughput"]
            ep_violations += info["violations"]
            state = next_state  
        results.append({
            "Method": name,
            "Avg Latency": ep_latency / 500,
            "Avg Throughput": ep_throughput / 500,
            "Total Violations": ep_violations,
            "Total Reward": ep_reward
        })
    df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(df.to_string(index=False))
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/comparison_results.csv", index=False)
    return df

if __name__ == "__main__":
    run_comparison()
