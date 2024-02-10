import sys
sys.path.append("NET")
from NET.Agent import Agent
import gym
from NET.transforms import  apply_wrappers
import numpy as np
import gc
import torch
from collections import deque

# Specify environment location
env_name = "BreakoutDeterministic-v4"
NUM_EPISODES = 45_000 
DISPLAY = True
LR = 1E-4
N_FRAMES = 4
SKIP_ACTIONS = 4
EPS_STRT = 1
EPS_MIN = .1
EPS_DEC = 0.99992
MIN_EPISODES_TO_LEARN = 5
UPDATE_FRAME_COUNT = 4


# Initialize Gym Environment
env =gym.make( 'ALE/Breakout-v5',render_mode='human' if DISPLAY else 'rgb_array' ) #render_mode="rgb_array")

# apply wrappers
env = apply_wrappers(env=env)

# Create an agent
agent = Agent(state_space=(4,84,84), action_space=4, model_name='32x64x64_breakout_model', gamma=.99,
                eps_strt=1., eps_min=.1, eps_dec=0.9992, batch_size=32, lr=.00025, number_frames=N_FRAMES)


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

    # Reset environment and preprocess state
    obs, _  = env.reset()
    state, _, _, _, _ = env.step(1)

    score = 0
            
    while not done:
         # Take epsilon greedy action
        action = agent.choose_action(state)

        next_state, reward, done, trunk, info = env.step(action)

        # clip reward
        reward = np.sign(reward)

        score += reward
        state = next_state


    scores.append(score)
    print(f'Episode {i}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}') 