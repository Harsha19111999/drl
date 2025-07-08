import gymnasium as gym
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation, MaxAndSkipObservation
import numpy as np
import imageio
from tqdm import tqdm
import ale_py

# Hyperparameters
ACTOR_LR, CRITIC_LR = 2.5e-4, 2.5e-4
GAMMA, LAMBDA = 0.99, 0.95
EPSCLIP, BETA = 0.1, 0.01
NUM_EPOCHS = 4
ROLLOUT_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Actor-Critic networks
class Actor(nn.Module):
    def __init__(self, num_actions=6):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.fc = nn.Linear(7*7*64, 512)
        self.policy = nn.Linear(512, num_actions)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return F.softmax(self.policy(x), dim=-1)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32,64,4,2)
        self.conv3 = nn.Conv2d(64,64,3,1)
        self.fc = nn.Linear(7*7*64, 512)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.value(x).squeeze(-1)

# 2. Wrapped environment
def make_env():
    gym.register_envs(ale_py)
    env = gym.make("PongDeterministic-v4", render_mode="rgb_array")
    env = MaxAndSkipObservation(env, skip=4)
    env = GrayscaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, shape=(84,84))
    env = FrameStackObservation(env, num_stack=4)
    return env

# 3. PPO Agent class
class PPOAgent:
    def __init__(self):
        self.actor = Actor().to(DEVICE)
        self.critic = Critic().to(DEVICE)
        self.opt_a = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.opt_c = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def sample_batch(self, env):
        obs, _ = env.reset()
        obs = torch.tensor(obs).float().permute(2,0,1).unsqueeze(0).to(DEVICE)
        buffer = {'obs':[], 'acts':[], 'logps':[], 'rews':[], 'vals':[]}
        for _ in range(ROLLOUT_LEN):
            with torch.no_grad():
                pi = self.actor(obs)
                v = self.critic(obs)
            dist = torch.distributions.Categorical(pi)
            a = dist.sample()
            logp = dist.log_prob(a)
            nxt, r, term, trunc, _ = env.step(a.item())
            buffer['obs'].append(obs)
            buffer['acts'].append(a)
            buffer['logps'].append(logp)
            buffer['rews'].append(r)
            buffer['vals'].append(v.item())
            if term or trunc:
                nxt, _ = env.reset()
            obs = torch.tensor(nxt).float().permute(2,0,1).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            buffer['vals'].append(self.critic(obs).item())
        return buffer

    def compute_gae(self, rews, vals):
        adv = np.zeros(len(rews), np.float32)
        last = 0
        for i in reversed(range(len(rews))):
            delta = rews[i] + GAMMA * vals[i+1] - vals[i]
            adv[i] = last = delta + GAMMA * LAMBDA * last
        ret = adv + np.array(vals[:-1])
        return adv, ret

    def update(self, buf):
        obs = torch.cat(buf['obs']).to(DEVICE)
        acts = torch.stack(buf['acts']).to(DEVICE)
        oldlog = torch.stack(buf['logps']).to(DEVICE)
        rews, vals = buf['rews'], buf['vals']
        adv, ret = self.compute_gae(rews, vals)
        adv = torch.tensor((adv - adv.mean())/(adv.std()+1e-8), device=DEVICE)
        ret = torch.tensor(ret, device=DEVICE)

        for _ in range(NUM_EPOCHS):
            pi = self.actor(obs)
            v = self.critic(obs)
            dist = torch.distributions.Categorical(pi)
            logp = dist.log_prob(acts)
            ratio = torch.exp(logp - oldlog)
            ent = dist.entropy().mean()
            clip = torch.clamp(ratio, 1-EPSCLIP, 1+EPSCLIP)
            loss_a = -(torch.min(ratio*adv, clip*adv)).mean() - BETA*ent
            loss_c = F.mse_loss(v, ret)
            self.opt_a.zero_grad(); loss_a.backward(); self.opt_a.step()
            self.opt_c.zero_grad(); loss_c.backward(); self.opt_c.step()
        return loss_a.item(), loss_c.item()

    def play(self, env, episodes=1, filename="pong.gif"):
        frames = []
        for _ in range(episodes):
            obs, _ = env.reset()
            done=False
            while not done:
                obs_t = torch.tensor(obs).float().permute(2,0,1).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    a = torch.argmax(self.actor(obs_t)).item()
                obs, _, done, _, _ = env.step(a)
                frames.append(obs[:,:,0])  # grayscale
        imageio.mimsave(filename, np.stack(frames), fps=30)

# 4. Training
if __name__ == "__main__":
    env = make_env()
    agent = PPOAgent()
    for it in tqdm(range(500)):
        buf = agent.sample_batch(env)
        la, lc = agent.update(buf)
        if it%50==0:
            print(f"Iter {it}, loss_a={la:.3f}, loss_c={lc:.3f}")
    agent.play(env)
