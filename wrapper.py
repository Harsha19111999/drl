import gymnasium as gym
from utils import *
import ale_py
import numpy as np
import torch

class GymWrapper():
    def __init__(self, env_name, history_length=4, obs_type="grayscale"):
        gym.register_envs(ale_py)
        self.env = gym.make(env_name, obs_type=obs_type)
        self.history_length = history_length
        self.state = None
        
    def reset(self):
        frame, info = self.env.reset()
        processed_image = process_image(frame)
        self.state = np.repeat(processed_image, self.history_length, axis=0)        
        return torch.tensor(self.state, dtype=torch.float32)
        
    def step(self, action):
        frame, reward, terminated, truncated, info = self.env.step(action)
        processed_image = process_image(frame)
        self.state = np.append(self.state[1:, :, :], processed_image, axis=0) 
        return torch.tensor(self.state, dtype=torch.float32), torch.tensor(reward, dtype=torch.float32), terminated or truncated
    
    def close(self):
        self.env.close()
