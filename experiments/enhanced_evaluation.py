import matplotlib.pyplot as plt
from env.network_env import NetworkSlicingEnv
from rl.agent import SlicingSAC
from baseline.static_allocation import StaticAllocation
from baseline.demand_based import DemandBasedAllocation
from rl.train_sac import train
import os
import numpy as np
import pandas as pd

def calculate_jain_fairness(allocations):
    """Calculates Jain's Fairness Index for resource allocation."""
    n = len(allocations)
    return (np.sum(allocations)**2) / (n * np.sum(allocations**2) + 1e-9)

def run_enhanced_evaluation(num_steps=500, stress_test=False):
    env = NetworkSlicingEnv()
    # Train or Load Agent
    print("Step 1: Training SAC Agent...")
    agent, _, _ = train(num_episodes=50, max_steps=500, save_curves=False)
    algorithms = {
        "Static": StaticAllocation(),
        "Demand-Based": DemandBasedAllocation(),
        "SAC (AI)": agent
    }
    all_metrics = {}
    for name, algo in algorithms.items():
        print(f"Step 2: Evaluating {name}...")
        state, _ = env.reset()
        metrics = {
            "step": [],
            "reward": [],
            "total_throughput": [],
            "total_latency": [],
            "total_violations": [],
            "embb_throughput": [],
            "urllc_latency": [],
            "mmtc_violations": [],
            "fairness_index": []
        }
        for step in range(num_steps):
            # Simulate Flash Crowd at step 250
            if stress_test and step == 250:
                print(f"  [!] Flash Crowd initiated at step {step}")
                for s_name in env.slice_names:
                    env.slices[s_name].traffic_demand *= 5.0
            if name == "SAC (AI)":
                action = algo.select_action(state, deterministic=True)
            else:
                action = algo.select_action(state)   
            next_state, reward, done, _, info = env.step(action)
            # Collect Metrics
            metrics["step"].append(step)
            metrics["reward"].append(reward)
            metrics["total_throughput"].append(info["throughput"])
            metrics["total_latency"].append(info["latency"])
            metrics["total_violations"].append(info["violations"])
            metrics["embb_throughput"].append(env.slices["eMBB"].throughput)
            metrics["urllc_latency"].append(env.slices["URLLC"].latency)
            metrics["mmtc_violations"].append(env.slices["mMTC"].qos_violations)
            # Calculate fairness based on PRB allocation ratios
            allocs = np.array(list(info["bw_alloc"].values()))
            metrics["fairness_index"].append(calculate_jain_fairness(allocs))
            state = next_state  
        all_metrics[name] = pd.DataFrame(metrics)
    # Visualization
    print("Step 3: Generating Professional KPI Plots...")
    os.makedirs("results/evaluation", exist_ok=True)
    # Plot 1: Per-Slice Critical KPIs
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for name, df in all_metrics.items():
        axes[0, 0].plot(df["step"], df["embb_throughput"], label=name)
        axes[0, 1].plot(df["step"], df["urllc_latency"], label=name)
        axes[1, 0].plot(df["step"], df["total_violations"].cumsum(), label=name)
        axes[1, 1].plot(df["step"], df["fairness_index"], label=name)
    axes[0, 0].set_title("eMBB Throughput (Mbps) - Higher is Better")
    axes[0, 1].set_title("URLLC Latency (ms) - Lower is Better")
    axes[1, 0].set_title("Cumulative QoS Violations - Lower is Better")
    axes[1, 1].set_title("Jain's Fairness Index - Closer to 1.0 is Better")
    for ax in axes.flat:
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("results/evaluation/slice_kpis.png")
    # Plot 2: Latency CDF (The "Wow" Plot for URLLC)
    plt.figure(figsize=(10, 6))
    for name, df in all_metrics.items():
        latencies = np.sort(df["urllc_latency"])
        cdf = np.arange(len(latencies)) / float(len(latencies))
        plt.plot(latencies, cdf, label=name, lw=2)
    plt.axvline(x=5.0, color='r', linestyle='--', label="URLLC Constraint (5ms)")
    plt.title("Latency CDF: Reliability Proof (URLLC)")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Probability (P_latency <= X)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/evaluation/latency_cdf.png")
    # Save CSV Summary
    summary = []
    for name, df in all_metrics.items():
        summary.append({
            "Method": name,
            "Avg eMBB Throughput": df["embb_throughput"].mean(),
            "Max URLLC Latency": df["urllc_latency"].max(),
            "Reliability (URLLC < 5ms)": (df["urllc_latency"] < 5.0).mean() * 100,
            "Total Violations": df["total_violations"].sum(),
            "Avg Fairness": df["fairness_index"].mean()
        })
    summary_df = pd.DataFrame(summary)
    print("\n--- PROFESSIONAL NETWORK AUDIT SUMMARY ---")
    print(summary_df.to_string(index=False))
    summary_df.to_csv("results/evaluation/summary_audit.csv", index=False)
    print("\nEvaluation complete. Plots saved to results/evaluation/")

if __name__ == "__main__":
    run_enhanced_evaluation(num_steps=500, stress_test=True)
