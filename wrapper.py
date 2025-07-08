import gymnasium as gym
from utils import *
import numpy as np
import torch
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
import ale_py

class GymWrapper():
    def __init__(self, env_name, history_length=4):
        self.history_length = history_length

        # Function to create a single wrapped environment
        def make_env():
            gym.register_envs(ale_py)
            base_env = gym.make(env_name, render_mode=None)
            env = MaxAndSkipEnv(base_env, skip=4)
            return env

        # Create vectorized environment (8 parallel envs)
        self.envs = gym.vector.SyncVectorEnv([make_env for _ in range(8)])
        self.state = None

    def reset(self):
        frame, info = self.envs.reset()
        processed_image = process_image(frame)
        self.state = np.repeat(processed_image, self.history_length, axis=1)        
        return torch.tensor(self.state, dtype=torch.float32)

    def step(self, action):
        frame, reward, terminated, truncated, info = self.envs.step(action)
        processed_image = process_image(frame)
        self.state = np.append(self.state[:, 1:, :, :], processed_image, axis=1) 
        done = np.logical_or(terminated, truncated)
        return torch.tensor(self.state, dtype=torch.float32), torch.tensor(reward, dtype=torch.float32), done

    def close(self):
        self.envs.close()
