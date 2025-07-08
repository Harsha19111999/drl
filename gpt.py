import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==== Utils ====

def process_image(frame, shape=(84, 84)):
    # Crop frame (remove score and borders)
    frame = frame[30:195, :, :]  # HWC
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Grayscale
    frame = frame.astype(np.uint8)
    return frame

# ==== Environment Wrapper ====

class GymWrapper:
    def __init__(self, env_name, history_length=4):
        self.env = gym.make(env_name)
        self.history_length = history_length
        self.state_buffer = []

    def reset(self):
        frame = self.env.reset()
        processed = process_image(frame)
        self.state_buffer = [processed for _ in range(self.history_length)]
        state = np.stack(self.state_buffer, axis=0)  # shape: (4, 84, 84)
        return state.astype(np.float32)

    def step(self, action):
        next_frame, reward, done, info = self.env.step(action)
        processed = process_image(next_frame)
        self.state_buffer.append(processed)
        if len(self.state_buffer) > self.history_length:
            self.state_buffer.pop(0)
        state = np.stack(self.state_buffer, axis=0)
        return state.astype(np.float32), reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

# ==== Actor-Critic Network ====

class ActorCritic(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(7*7*64, 512)

        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        # x: (batch, 4, 84, 84)
        x = x / 255.0  # Normalize pixel values
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))

        logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return logits, value

# ==== PPO Agent ====

class PPOAgent:
    def __init__(self, env, device, gamma=0.99, lam=0.95, clip_epsilon=0.1,
                 vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=2.5e-4, epochs=4, batch_size=64):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size

        self.ac = ActorCritic(env.env.action_space.n).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

    def compute_gae(self, rewards, masks, values):
        advantages = torch.zeros_like(rewards).to(self.device)
        last_adv = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * masks[t] - values[t]
            advantages[t] = delta + self.gamma * self.lam * masks[t] * last_adv
            last_adv = advantages[t]
        returns = advantages + values[:-1]
        return advantages, returns

    def collect_trajectories(self, rollout_length=2048):
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []

        state = self.env.reset()
        done = False

        for _ in range(rollout_length):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, value = self.ac(state_tensor)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, done, _ = self.env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            dones.append(1 - int(done))
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state

            if done:
                state = self.env.reset()

        # Add value for the last next state
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, last_value = self.ac(state_tensor)
        values.append(last_value.item())

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)  # (rollout_length, 4, 84, 84)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        log_probs = torch.tensor(log_probs).to(self.device)
        values = torch.tensor(values).to(self.device)

        return states, actions, rewards, dones, log_probs, values

    def update(self, states, actions, rewards, dones, old_log_probs, values, iteration, max_iterations):
        advantages, returns = self.compute_gae(rewards, dones, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_samples = states.shape[0]
        for epoch in range(self.epochs):
            indices = torch.randperm(num_samples)
            for start in range(0, num_samples, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]

                batch_states = states[batch_idx].to(self.device)
                batch_actions = actions[batch_idx].to(self.device)
                batch_old_log_probs = old_log_probs[batch_idx].to(self.device)
                batch_advantages = advantages[batch_idx].to(self.device)
                batch_returns = returns[batch_idx].to(self.device)

                logits, values_pred = self.ac(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - batch_old_log_probs)
                alpha = 1.0 - iteration / max_iterations
                clip_range = self.clip_epsilon * alpha

                surr1 = batch_advantages * ratio
                surr2 = batch_advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = batch_returns + (values_pred - batch_returns).clamp(-clip_range, clip_range)
                value_losses = (values_pred - batch_returns).pow(2)
                value_losses_clipped = (value_pred_clipped - batch_returns).pow(2)
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return advantages.mean().item(), entropy.item(), policy_loss.item(), value_loss.item(), loss.item()

# ==== Training Loop ====

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GymWrapper("BreakoutNoFrameskip-v4")
    agent = PPOAgent(env, device)

    max_updates = 40000
    rollout_length = 2048
    batch_size = 64
    log_interval = 10

    avg_rewards = []

    for update in tqdm(range(max_updates)):
        states, actions, rewards, dones, log_probs, values = agent.collect_trajectories(rollout_length=rollout_length)
        adv_mean, entropy, p_loss, v_loss, total_loss = agent.update(
            states, actions, rewards, dones, log_probs, values,
            iteration=update, max_iterations=max_updates
        )

        total_reward = rewards.sum().item() / rollout_length
        avg_rewards.append(total_reward)

        if update % log_interval == 0:
            print(f"Update {update} | Avg Reward: {total_reward:.3f} | Advantage Mean: {adv_mean:.3f} | Entropy: {entropy:.3f} | Policy Loss: {p_loss:.3f} | Value Loss: {v_loss:.3f}")

            plt.figure(figsize=(10,5))
            plt.plot(avg_rewards)
            plt.title("Average Reward Over Time")
            plt.xlabel("Updates")
            plt.ylabel("Average Reward")
            plt.grid()
            plt.pause(0.01)
            plt.close()

    env.close()

if __name__ == "__main__":
    train()
