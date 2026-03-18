import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from .model import ContinuousActor, QuantileCritic
from .utils import quantile_huber_loss
from .lagrangian import PIDLagrangian, ConstraintSpec

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done, cost_dict=None):
        self.buffer.append((state, action, reward, next_state, done, cost_dict or {}))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, costs = zip(*batch)
        # convert cost list to tensor format
        cost_tensor_dict = {}
        if costs and costs[0]:
            for key in costs[0].keys():
                cost_tensor_dict[key] = torch.FloatTensor(np.array([c[key] for c in costs]))     
        return (
            torch.FloatTensor(np.array(state)),
            torch.FloatTensor(np.array(action)),
            torch.FloatTensor(np.array(reward)).unsqueeze(1),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(np.array(done)).unsqueeze(1),
            cost_tensor_dict
        )
    def __len__(self):
        return len(self.buffer)

class SlicingSAC:
    def __init__(self, obs_dim, action_dim, device="cpu"):
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=1e-4) # reduced lr
        # URLLC Latency should be <= 5ms
        self.lagrangian = PIDLagrangian([
            ConstraintSpec(name="urllc_latency", target=5.0, kp=0.1, ki=0.01)
        ])
        # Networks
        self.actor = ContinuousActor(obs_dim, action_dim).to(self.device)
        self.critic = QuantileCritic(obs_dim, action_dim).to(self.device)
        self.critic_target = QuantileCritic(obs_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        # Optimizers (reduced lr for stability)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        # specific to quantile SAC
        self.n_quantiles = 32
        self.taus = torch.linspace(1.0/(2*self.n_quantiles), 1 - 1.0/(2*self.n_quantiles), self.n_quantiles).to(self.device)
    
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if deterministic:
            action = self.actor.deterministic(state)
        else:
            action, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]
    
    def update(self, replay_buffer, batch_size):
        state, action, reward, next_state, done, cost_dict = replay_buffer.sample(batch_size)
        state, action = state.to(self.device), action.to(self.device)
        reward, next_state, done = reward.to(self.device), next_state.to(self.device), done.to(self.device)
        # Update Lagrangian
        lagrange_stats = {}
        if cost_dict:
            cost_dict = {k: v.to(self.device) for k, v in cost_dict.items()}
            # r_eff = r - sum(lambda * (cost - target)_+)
            penalty = self.lagrangian.compute_penalty(cost_dict).unsqueeze(1)
            reward = reward - penalty
            lagrange_stats = self.lagrangian.update(cost_dict)
        # Update Alpha
        new_action, log_prob = self.actor.sample(state)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        # Update Critic
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_target, q2_target = self.critic_target(next_state, next_action)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_prob.unsqueeze(-1)
            target = reward + (1 - done) * self.gamma * q_target # [B, Nq]
        q1, q2 = self.critic(state, action)
        critic_loss = quantile_huber_loss(q1, target, self.taus) + quantile_huber_loss(q2, target, self.taus)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0) # grad clipping
        self.critic_optimizer.step()
        # Update Actor
        q1_pi, q2_pi = self.critic(state, new_action)
        q_pi = torch.min(q1_pi, q2_pi).mean(dim=-1)
        actor_loss = (self.alpha * log_prob - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0) # grad clipping
        self.actor_optimizer.step()
        # Soft Update Target Critic
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        out = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item()
        }
        out.update(lagrange_stats)
        return out
