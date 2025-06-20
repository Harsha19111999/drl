import os

# Name of the Gym environment for the agent to learn & play
ENV_NAME = 'Breakout-v4'

INPUT_SHAPE = (84, 84, 4)
BATCH_SIZE = 64
N_UPDATES = 1000000
HIDDEN = 1024
LR = 1e-3
GAMMA = 0.99
VALUE_C = 0.5
STD_ADV = True
AGENT = "PPO"
NUM_EPOCHS = 4

PATH_SAVE_MODEL = f"../model/{AGENT.lower()}"

PATH_LOAD_MODEL = ""
#PATH_LOAD_MODEL = None

CONFIG_WANDB = dict (
    learning_rate = LR,
    batch_size = BATCH_SIZE,
    agent = AGENT,
    operating_system = os.name
)
