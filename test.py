from network import ActorCritic
from wrapper import GymWrapper
from agent import Agent
from tqdm import tqdm
import torch
from config import *
import numpy as np
import matplotlib.pyplot as plt

# Initialize environment, networks, agent, and optimizers
pong = GymWrapper('BreakoutNoFrameskip-v4')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
ac = ActorCritic().to(device)
agent = Agent(ac, device)

advantages_list = []
entropy_list = []
log_probs_list = []
ratio_list = []
clipped_ratio_list = []
actor_loss_list = []
critic_loss_list = []
total_loss_list = []

# agent.play(pong, action_type="random")

# breakpoint()
# PPO training loop
for update in tqdm(range(1)):
    # Collect trajectories
    states, actions, rewards, log_probs, values = agent.collect_trajectories(pong, n_steps=2048)
    # Compute advantages and returns
    advantages, returns = agent.compute_gae(rewards, values, lam=0.95, gamma=0.99)
    # Train the actor-critic networks
    total_reward = sum(rewards)/len(rewards)

    advantages, entropy, log_probs, ratio, clipped_ratio, actor_loss, critic_loss, total_loss = agent.train(states, actions, log_probs, advantages, returns)
    advantages_list.append(advantages)
    entropy_list.append(entropy)
    log_probs_list.append(log_probs)
    ratio_list.append(ratio)
    clipped_ratio_list.append(clipped_ratio)
    actor_loss_list.append(actor_loss)
    critic_loss_list.append(critic_loss)
    total_loss_list.append(total_loss)

    print("Total reward: ", total_reward.mean().item())

# agent.play(pong, action_type="policy")

fig, axs = plt.subplots(2, 4, figsize=(20, 8))
axs = axs.flatten()  # Flatten to simplify indexing

# Metric titles and data
metrics = [
    ("Advantages", advantages_list),
    ("Entropy", entropy_list),
    ("Log Probs", log_probs_list),
    ("Ratio", ratio_list),
    ("Clipped Ratio", clipped_ratio_list),
    ("Actor Loss", actor_loss_list),
    ("Critic Loss", critic_loss_list),
    ("Total Loss", total_loss_list),
]

# Plot each metric
for i, (title, data) in enumerate(metrics):
    axs[i].plot(data, label=title, color='tab:blue')
    axs[i].set_title(title)
    axs[i].set_xlabel("Training Steps")
    axs[i].set_ylabel(title)
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()