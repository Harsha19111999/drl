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
losses = []

# agent.play(pong, action_type="random")

# breakpoint()
# PPO training loop
for update in tqdm(range(500)):
    # Collect trajectories
    states, actions, rewards, log_probs, values = agent.collect_trajectories(pong, n_steps=2048)
    # Compute advantages and returns
    advantages, returns = agent.compute_gae(rewards, values, lam=0.95, gamma=0.99)
    # Train the actor-critic networks
    total_reward = sum(rewards)/len(rewards)

    loss = agent.train(states, actions, log_probs, advantages, returns)
    losses.append(loss)
    # Log and visualize
    if update % 10 == 0:
        print(f"=== Update {update} ===")
        print("Total reward: ", total_reward.mean())

agent.play(pong, action_type="policy")

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Actor-Critic Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Actor-Critic Loss over Time')
plt.legend()
plt.grid()
plt.show()