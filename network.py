import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
'''
The input to the neural network consists is an 84 × 84 × 4 image produced by φ. The first hidden layer convolves 16 8 × 8 filters with stride 4 with the input image 
and applies a rectifier nonlinearity [10, 18]. The second hidden layer convolves 32 4 × 4 filters with stride 2, again followed by a rectifier nonlinearity. The
final hidden layer is fully-connected and consists of 256 rectifier units. The output layer is a fully connected linear layer with a single output for each valid action
'''

class ActorCritic(nn.Module):
    def __init__(self, input_channels=4, num_actions=4):  # 4 actions for Breakout
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7*7*64, 512)
        
        self.actor = nn.Linear(512, num_actions)
        self.critic = nn.Linear(512, 1)
        
    def forward(self, x):
        x = x / 255.0 
        x = F.relu(self.conv1(x))  # -> (B, 32, 20, 20)
        x = F.relu(self.conv2(x))  # -> (B, 64, 9, 9)
        x = F.relu(self.conv3(x))  # -> (B, 64, 7, 7)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc(x))     # Shared dense layer
        # print(torch.clamp(self.actor(x), -10, 10))
        return F.softmax(self.actor(x), dim=-1), self.critic(x).squeeze(-1) # Make sure softmax dims are correct
        
