import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from env.network_env import NetworkSlicingEnv
from rl.agent import SlicingSAC, ReplayBuffer

def train(num_episodes=200, max_steps=1000, save_curves=True):
    env = NetworkSlicingEnv()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SlicingSAC(obs_dim=obs_dim, action_dim=action_dim)
    replay_buffer = ReplayBuffer(capacity=100000)
    batch_size = 64
    rewards_history = []
    violations_history = []
    print(f"Starting Training ({num_episodes} episodes)...")
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_violations = 0
        for step in range(max_steps):
            if len(replay_buffer) < batch_size * 2:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            # cost
            cost_dict = {
                "urllc_latency": info["slice_latencies"]["URLLC"],
                "embb_violation": float(env.slices["eMBB"].qos_violations),
                "mmtc_violation": float(env.slices["mMTC"].qos_violations)
            }
            replay_buffer.push(state, action, reward, next_state, done, cost_dict=cost_dict)
            state = next_state
            episode_reward += reward
            episode_violations += info["violations"]
            if len(replay_buffer) > batch_size * 2:
                train_info = agent.update(replay_buffer, batch_size)
                if step % 100 == 0:
                    l_val = train_info.get("lambda_urllc_latency", 0)
                    print(f"  Step {step}: [Lambda_URLLC: {l_val:.4f}]", end="\r")
            if done:
                break
        rewards_history.append(episode_reward)
        violations_history.append(episode_violations)
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, Violations = {episode_violations}")
        # Periodic save
        if (episode + 1) % 10 == 0:
            os.makedirs("results", exist_ok=True)
            torch.save(agent.actor.state_dict(), "results/actor_weights.pth")
            torch.save(agent.critic.state_dict(), "results/critic_weights.pth")
            print(f"  Saved weights at episode {episode}")
    if save_curves:
        # Save results
        os.makedirs("results", exist_ok=True)
        
        # Save raw history
        df_hist = pd.DataFrame({
            "episode": range(num_episodes),
            "reward": rewards_history,
            "violations": violations_history
        })
        df_hist.to_csv("results/training_history.csv", index=False)
        
        # Professional plotting
        plt.style.use('seaborn-v0_8-muted')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f"Training Convergence ({num_episodes} Episodes)", fontsize=18, fontweight='bold')

        # Reward History
        ax1.plot(rewards_history, color='#4E79A7', linewidth=1.5)
        ax1.set_title("Cumulative Reward per Episode", fontsize=14, fontweight='semibold')
        ax1.set_xlabel("Episode", fontsize=12)
        ax1.set_ylabel("Total Reward", fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.6)

        # Violation History
        ax2.plot(violations_history, color='#E15759', linewidth=1.5)
        ax2.set_title("QoS Violations per Episode", fontsize=14, fontweight='semibold')
        ax2.set_xlabel("Episode", fontsize=12)
        ax2.set_ylabel("Violation Count", fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("results/Training Curves.png", dpi=300)
        print("Training Complete. Beautiful curves saved to results/Training Curves.png")
        # Save model
        torch.save(agent.actor.state_dict(), "results/actor_weights.pth")
        torch.save(agent.critic.state_dict(), "results/critic_weights.pth")
        print("Training Complete. Model and curves saved to results/")
    return agent, rewards_history, violations_history

if __name__ == "__main__":
    train()
