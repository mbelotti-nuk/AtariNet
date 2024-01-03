import torch
import gym
import os


if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

ENV_NAME = 'SuperMarioBros-1-1-v0'
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 5000
NUM_OF_EPISODES = 50_000

env =gym.make( 'ALE/Breakout-v5', render_mode="rgb_array")


# s

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)