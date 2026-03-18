import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env.network_env import NetworkSlicingEnv
from rl.agent import SlicingSAC
from baseline.static_allocation import StaticAllocation
from baseline.demand_based import DemandBasedAllocation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
STEPS = 500

def get_agent():
    env = NetworkSlicingEnv()
    agent = SlicingSAC(obs_dim=env.observation_space.shape[0], 
                       action_dim=env.action_space.shape[0])
    # load weights from results folder
    path = "results/actor_weights.pth"
    if os.path.exists(path):
        agent.actor.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"Loaded weights from {path}")
    else:
        print("No weights found!!")
    return agent

def run_test(name, algo, env):
    s, _ = env.reset()
    res = {"lat": [], "tp": [], "vio": [], "rew": []}
    for i in range(STEPS):
        if name == "SAC (AI)":
            a = algo.select_action(s, deterministic=True)
        else:
            a = algo.select_action(s)    
        next_s, r, done, _, info = env.step(a)
        res["lat"].append(info["latency"])
        res["tp"].append(info["throughput"])
        res["vio"].append(info["violations"])
        res["rew"].append(r)
        s = next_s
    return res

def main():
    env = NetworkSlicingEnv()
    sac = get_agent()
    algos = {
        "Static": StaticAllocation(),
        "Demand-Based": DemandBasedAllocation(),
        "SAC (AI)": sac
    }
    all_data = {}
    for n, al in algos.items():
        print(f"Testing {n}...")
        all_data[n] = run_test(n, al, env)  
    # Plotting results with high aesthetics
    plt.style.use('seaborn-v0_8-muted')
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("5G Network Slicing Performance Comparison", fontsize=20, fontweight='bold', y=0.98)
    
    colors = ['#4E79A7', '#F28E2B', '#E15759'] # Professional palette
    
    # Throughput Plot
    ax1 = axes[0, 0]
    for i, (n, d) in enumerate(all_data.items()):
        ax1.plot(d["tp"], label=n, color=colors[i], linewidth=2, alpha=0.8)
    ax1.set_title("Network Throughput (Mbps)", fontsize=14, fontweight='semibold')
    ax1.set_xlabel("Time Step", fontsize=12)
    ax1.set_ylabel("Mbps", fontsize=12)
    ax1.legend(frameon=True, shadow=True)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Latency Plot
    ax2 = axes[0, 1]
    for i, (n, d) in enumerate(all_data.items()):
        ax2.plot(d["lat"], label=n, color=colors[i], linewidth=2, alpha=0.8)
    ax2.set_title("Average Latency (ms)", fontsize=14, fontweight='semibold')
    ax2.set_xlabel("Time Step", fontsize=12)
    ax2.set_ylabel("ms", fontsize=12)
    ax2.legend(frameon=True, shadow=True)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Violations Bar
    ax3 = axes[1, 0]
    names = list(all_data.keys())
    vios = [sum(all_data[x]["vio"]) for x in names]
    bars = ax3.bar(names, vios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_title("Total QoS Violations", fontsize=14, fontweight='semibold')
    ax3.set_ylabel("Count", fontsize=12)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Avg Performance Bar
    ax4 = axes[1, 1]
    avg_tp = [np.mean(all_data[x]["tp"]) for x in names]
    bars_tp = ax4.bar(names, avg_tp, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    ax4.set_title("Average System Throughput", fontsize=14, fontweight='semibold')
    ax4.set_ylabel("Mbps", fontsize=12)
    # Add value labels
    for bar in bars_tp:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/comparison_plots.png", dpi=300)
    print("Saved high-resolution plots to results/comparison_plots.png")
    # save csv summary
    summary = []
    for n in names:
        summary.append({
            "Method": n,
            "Avg_Lat": np.mean(all_data[n]["lat"]),
            "Avg_TP": np.mean(all_data[n]["tp"]),
            "Total_Vio": sum(all_data[n]["vio"])
        })
    df = pd.DataFrame(summary)
    df.to_csv("results/comparison_results.csv", index=False)
    print(df)

if __name__ == "__main__":
    main()
