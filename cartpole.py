import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# --- Hyperparameters ---
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
BATCH_SIZE = 2048
EPOCHS = 4
ENV_NAME = "CartPole-v1"
N_UPDATES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Actor-Critic Networks ---
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
    
    def forward(self, x):
        return F.softmax(self.net(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)

# --- PPO Agent ---
class PPOAgent:
    def __init__(self, obs_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        probs = self.actor(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()
    
    def compute_gae(self, rewards, values, dones, last_value):
        values = values + [last_value]
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * LAMBDA * (1 - dones[i]) * gae
            returns.insert(0, gae + values[i])
        advs = np.array(returns) - np.array(values[:-1])
        return returns, advs

    def train(self, batch):
        states, actions, old_log_probs, returns, advs = batch
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions).to(device)
        old_log_probs = torch.tensor(old_log_probs).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advs = torch.tensor(advs, dtype=torch.float32).to(device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        for _ in range(EPOCHS):
            probs = self.actor(states)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            entropy = dist.entropy().mean()

            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advs
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

            values = self.critic(states)
            critic_loss = F.mse_loss(values, returns)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

import time

def test(agent, episodes=5):
    env = gym.make(ENV_NAME, render_mode="human")
    for ep in range(episodes):
        obs = env.reset()[0]
        done = False
        total_reward = 0
        while not done:
            time.sleep(0.02)  # Slow down rendering for visibility
            state = torch.tensor(obs, dtype=torch.float32).to(device)
            probs = agent.actor(state)
            action = torch.argmax(probs).item()  # Deterministic
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Test Episode {ep+1}: Reward = {total_reward}")
    env.close()


# --- Main Training Loop ---
def train():
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    agent = PPOAgent(obs_dim, act_dim)

    for update in range(N_UPDATES):
        obs = env.reset()[0]
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        total_reward = 0

        for _ in range(BATCH_SIZE):
            action, log_prob, _ = agent.select_action(obs)
            value = agent.critic(torch.tensor(obs, dtype=torch.float32).to(device)).item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value)

            obs = next_obs
            total_reward += reward
            if done:
                obs = env.reset()[0]

        last_value = agent.critic(torch.tensor(obs, dtype=torch.float32).to(device)).item()
        returns, advs = agent.compute_gae(rewards, values, dones, last_value)
        batch = (states, actions, log_probs, returns, advs)
        agent.train(batch)

        avg_reward = np.sum(rewards) / (np.sum(dones) + 1e-8)
        print(f"Update {update}: Avg Episode Reward = {avg_reward:.2f}")

    env.close()
    print("Training complete. Running test episodes...")
    test(agent)



if __name__ == "__main__":
    train()
   

