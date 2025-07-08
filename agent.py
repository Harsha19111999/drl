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
        breakpoint()
        total_reward = 0
        cv2.namedWindow('Breakout Agent', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Breakout Agent', 800, 800)
        while True:
            if action_type == "policy": 
                with torch.no_grad():
                    policy, _ = self.ac(state)
                    action = torch.argmax(policy, axis=1)
                    # print(action)
                next_state, reward, done = wrapper.step(action)
            elif action_type == "random":
                next_state, reward, done = wrapper.step(wrapper.envs.action_space.sample())

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
    
    def save_states_gif(self, states_list, save_path="trajectories.gif"):
        """
        states_list: tensor of shape [T, num_envs, C, H, W]
        save_path: path to save gif
        """
        # Move to CPU and convert to numpy
        states_np = states_list.cpu().numpy()
        T, num_envs, C, H, W = states_np.shape

        frames = []

        for t in range(T):
            # Collect each env's last channel frame
            env_frames = []
            for env_idx in range(num_envs):
                frame = states_np[t, env_idx, -1, :, :]  # Take last channel (assuming grayscale)
                frame = (frame * 255).astype(np.uint8)   # Convert to 0–255 if not already
                env_frames.append(frame)

            # Concatenate all env frames horizontally
            combined_frame = np.concatenate(env_frames, axis=1)  # Shape: (H, num_envs * W)

            # Convert single-channel to RGB for GIF
            combined_frame_rgb = np.stack([combined_frame]*3, axis=-1)  # Shape: (H, W_total, 3)

            frames.append(combined_frame_rgb)

        # Write GIF
        imageio.mimsave(save_path, frames, fps=5)
        print(f"Saved GIF to {save_path}")
    
    def collect_trajectories(self, wrapper, n_steps=200):
        states_list = []
        actions_list = []
        rewards_list = []
        log_probs_list = []
        values_list = []

        state = wrapper.reset().to(self.device)
        for _ in range(n_steps):
            with torch.no_grad():
                probs, value = self.ac(state)
            distribution = torch.distributions.Categorical(probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            next_state, reward, done = wrapper.step(action)
            
            states_list.append(state)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)
            values_list.append(value)

            state = next_state.to(self.device)

            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if done.any():
                state = wrapper.reset().to(self.device)
                fire_action = torch.tensor([1]*8).to(self.device)
                state, _, _ = wrapper.step(fire_action)
                state = state.to(self.device)

        with torch.no_grad():
            _, final_value = self.ac(state)
        values_list.append(final_value)
        states_list = torch.stack(states_list).to(self.device)
        actions_list = torch.stack(actions_list).to(self.device)
        rewards_list = torch.stack(rewards_list).to(self.device)
        log_probs_list = torch.stack(log_probs_list).to(self.device)
        values_list = torch.stack(values_list).to(self.device)

        return states_list, actions_list, rewards_list, log_probs_list, values_list

    def compute_gae(self, rewards, values, lam, gamma):
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        for i in reversed(range(len(rewards))): 
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            advantages[i] = last_gae = delta + gamma * lam * last_gae
        returns = advantages + values[:-1]
        # breakpoint()
        return advantages, returns  

    def compute_entropy(self, probs):
        # breakpoint()
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
    
    def compute_log_probs(self, probs, actions):
        # The below two lines are equivalent to this: torch.log(probs.gather(1, actions.unsqueeze(1)).squeeze(1))
        dist = torch.distributions.Categorical(probs)
        return dist.log_prob(actions)

    def train(self, old_states, old_actions, old_log_probs, advantages, returns, epsilon=0.1, beta=0.01, c1=1):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # total_loss = 0
        dataset_size = old_states.shape[0]
        batch_size = 256  # Feel free to tune this
        
        for _ in range(NUM_EPOCHS):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                minibatch_idx = indices[start:end]

                old_states_mb = old_states[minibatch_idx]
                old_actions_mb = old_actions[minibatch_idx]
                old_log_probs_mb = old_log_probs[minibatch_idx]
                returns_mb = returns[minibatch_idx]
                advantages_mb = advantages[minibatch_idx]

                old_states_mb = old_states_mb.reshape(-1, *old_states_mb.shape[2:])
                old_actions_mb = old_actions_mb.reshape(-1)
                old_log_probs_mb = old_log_probs_mb.reshape(-1)
                returns_mb = returns_mb.reshape(-1)
                
                advantages_mb = advantages_mb.reshape(-1)
                probs, values = self.ac(old_states_mb)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(old_actions_mb)
                entropy = (dist.entropy()).mean()
                # breakpoint()
                # entropy = self.compute_entropy(probs)
                # log_probs = self.compute_log_probs(probs, old_actions_mb)
                # PPO clipped surrogate loss
                ratio = torch.exp(log_probs - old_log_probs_mb)
                clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
                actor_loss = -torch.min(ratio * advantages_mb, clipped_ratio * advantages_mb).mean()

                # Critic loss
                critic_loss = F.mse_loss(values.squeeze(), returns_mb)
                # breakpoint()

                # Combined loss
                total_loss = actor_loss + c1 * critic_loss - beta * entropy

                # breakpoint()
                self.ac_optimizer.zero_grad()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
                self.ac_optimizer.step()
        print(returns_mb.mean())
        print("Mean AC loss:", total_loss.item())
        return advantages_mb.mean().item(), entropy.item(), log_probs.mean().item(), ratio.mean().item(), clipped_ratio.mean().item(), actor_loss.item(), critic_loss.item(), total_loss.item() 