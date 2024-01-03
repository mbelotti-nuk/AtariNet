import gym
import random
import time
from gym.utils.play import play
import numpy as np

env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
play(env=env, zoom=3)

