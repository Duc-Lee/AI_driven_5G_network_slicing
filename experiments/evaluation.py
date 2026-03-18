import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_results(csv_path="results/comparison_results.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return    
    df = pd.read_csv(csv_path)
    os.makedirs("results", exist_ok=True)
    # Latency Comparison
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.barplot(x="Method", y="Avg Latency", data=df, palette="viridis")
    plt.title("Average Latency (ms)")
    plt.ylabel("Latency")
    # Throughput Comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x="Method", y="Avg Throughput", data=df, palette="magma")
    plt.title("Average Throughput (Mbps)")
    plt.ylabel("Throughput")
    # QoS Violations Comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x="Method", y="Total Violations", data=df, palette="rocket")
    plt.title("Total QoS Violations")
    plt.ylabel("Violations")
    # otal Reward Comparison
    plt.subplot(2, 2, 4)
    sns.barplot(x="Method", y="Total Reward", data=df, palette="crest")
    plt.title("Total Reward")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("results/comparison_plots.png")
    print("Plots saved to results/comparison_plots.png")

if __name__ == "__main__":
    plot_results()
