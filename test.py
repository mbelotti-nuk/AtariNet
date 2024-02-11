import sys
sys.path.append("NET")
from NET.Agent import Agent
import gym
from NET.transforms import Environment
import numpy as np
import gc
import torch
from collections import deque

# Specify environment location
env_name = "ALE/Breakout-v5"
NUM_EPISODES = 200 
DISPLAY = True

N_FRAMES = 4
SKIP_ACTIONS = 4

UPDATE_FRAME_COUNT = 4


# Initialize Gym Environment
env =gym.make( env_name, render_mode='human' if DISPLAY else 'rgb_array' )

ENVIRONMENT = Environment(env, skip_frames=SKIP_ACTIONS, n_frames=N_FRAMES)
# Create an agent
agent = Agent(state_space=(4,84,84), action_space=4, 
              model_name='32x64x64_breakout_model', gamma=.99,
               batch_size=32, lr=5E-5)


# Clean environment
if torch.cuda.is_available():
    # Empty cache in cuda if needed
    torch.cuda.empty_cache()
    # Garbage collect
    gc.collect()

scores = []
max_score = 0


for i in range(NUM_EPISODES):
    done = False
    # re-set the environment
    ENVIRONMENT.reset()
    score = 0
            
    while not done:
         # Take only greedy action
        action = agent.choose_action(ENVIRONMENT.state(), train=False)

        # make a step
        state, next_state, action, reward, done = ENVIRONMENT.step(action)

        score += reward


    scores.append(score)
    print(f'Episode {i}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}') 