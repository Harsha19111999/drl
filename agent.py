from config import *
from tqdm import tqdm
import torch
import numpy as np 
import torch.nn.functional as F
import imageio
import torch.optim as optim
from wrapper import *
import cv2 

class Agent():
    def __init__(self, ac, device):
        self.ac = ac.to(device)
        self.ac_optimizer = optim.Adam(self.ac.parameters(), lr=LR)
        self.device = device
        
    def play(self, wrapper, action_type):
        state = wrapper.reset().to(self.device)
        total_reward = 0
        cv2.namedWindow('Breakout Agent', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Breakout Agent', 800, 800)
        while True:
            if action_type == "policy": 
                with torch.no_grad():
                    policy, _ = self.ac(state.unsqueeze(0))
                    action = torch.argmax(policy).item()
                    # print(action)
                next_state, reward, done = wrapper.step(action)
            elif action_type == "random":
                next_state, reward, done = wrapper.step(wrapper.env.action_space.sample())

            total_reward += reward
            cv2.imshow('Breakout Agent', state[:1, :, :].cpu().numpy().reshape((84, 84)))
            key = cv2.waitKey(30)  # Display each frame for 30ms (~33fps)
            if key == ord('q'):
                break
            state = next_state.to(self.device)

            if done:
                break
        
        print("The total reward obtained is:", total_reward)
        wrapper.close()
        cv2.destroyAllWindows()
        # stacked = np.concatenate(animation_frames, axis=0)
        # print(stacked.shape)
        # imageio.mimsave('output.gif', stacked, fps=1)
    
    def collect_trajectories(self, wrapper, n_steps=200):
        states_list = []
        actions_list = []
        rewards_list = []
        log_probs_list = []
        values_list = []

        state = wrapper.reset().to(self.device)
        for _ in range(n_steps):
            state_tensor = state.unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                probs, value = self.ac(state_tensor)

            distribution = torch.distributions.Categorical(probs)
            action = distribution.sample()
            log_prob = torch.log(probs.squeeze(0)[action])
            next_state, reward, done = wrapper.step(action.item())

            states_list.append(state)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            values_list.append(value.item())

            state = next_state.to(self.device)

            if done:
                state = wrapper.reset().to(self.device)
                break

        with torch.no_grad():
            _, final_value = self.ac(state.unsqueeze(0))
            final_value = final_value.item()
        values_list.append(final_value)

        return states_list, actions_list, rewards_list, log_probs_list, values_list

    def compute_gae(self, rewards, values, lam, gamma):
        # print(rewards)
        # print(values)
        # breakpoint()
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0
        for i in reversed(range(len(rewards))): 
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            advantages[i] = last_gae = delta + gamma * lam * last_gae
        returns = advantages + np.array(values[:-1])
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return advantages, returns  

    def compute_entropy(self, probs):
        return -torch.sum(probs * torch.log(probs + 1e-8))
    
    def compute_log_probs(self, probs, actions):
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)

    def train(self, old_states, old_actions, old_log_probs, advantages, returns, epsilon=0.2, beta=0.01, c1=0.5):
        old_states = torch.stack(old_states).to(self.device)            # Shape: (T, C, H, W)
        old_actions = torch.stack(old_actions).to(self.device)          # Shape: (T,)
        old_log_probs = torch.stack(old_log_probs).to(self.device)      # Shape: (T,)
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # total_loss = 0
        dataset_size = old_states.shape[0]
        batch_size = 32  # Feel free to tune this
        for _ in range(NUM_EPOCHS):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                minibatch_idx = indices[start:end]

                states_mb = old_states[minibatch_idx]
                actions_mb = old_actions[minibatch_idx]
                old_log_probs_mb = old_log_probs[minibatch_idx]
                returns_mb = returns[minibatch_idx]
                advantages_mb = advantages[minibatch_idx]

                probs, values = self.ac(states_mb)
                # dist = torch.distributions.Categorical(probs)
                # log_probs = dist.log_prob(actions_mb)
                # entropy = dist.entropy().mean()
                entropy = self.compute_entropy(probs)
                log_probs = self.compute_log_probs(probs, actions_mb)
                # PPO clipped surrogate loss
                ratio = torch.exp(log_probs - old_log_probs_mb)
                clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                actor_loss = -torch.min(ratio * advantages_mb, clipped_ratio * advantages_mb).mean()

                # Critic loss
                critic_loss = F.mse_loss(values.squeeze(), returns_mb)

                # Combined loss
                total_loss = actor_loss + c1 * critic_loss - beta * entropy

                self.ac_optimizer.zero_grad()
                total_loss.backward()
                self.ac_optimizer.step()
        print("Mean AC loss:", total_loss.item())
        return total_loss.item()