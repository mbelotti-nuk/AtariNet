import sys
sys.path.append("NET")
from NET.Agent import Agent
import gym
from NET.transforms import Transforms
import numpy as np
import gc
import torch
from collections import deque

# Specify environment location
env_name = 'ALE/Breakout-v5'
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
    state = Transforms.to_gray(obs[0])
    state = deque( [state[0]]*N_FRAMES, maxlen=N_FRAMES )


    n_lifes = 5
    score = 0
    frame_count = 0
            
    while not done:
        # Take epsilon greedy action
        action = agent.choose_action(state, train=False)

        new_state = []
        new_observations = []
        observations = []
        action_reward = 0

        # take next actions: the number of consecutive actions is qeual to SKIP_ACTIONS
        # check: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        obs_stack = deque(maxlen=2)
        obs_stack.append(obs[0])
        for k in range(SKIP_ACTIONS):
            obs_, reward, done, trunc, info = env.step(action)
            action_reward += reward
            obs_stack.append(obs_)
            if done: break

        # blur last two frames
        obs_ = np.maximum(*list(obs_stack))
        # transform the observation
        state_ = Transforms.to_gray(obs_).squeeze(0)

        # next state
        new_state = state.copy()
        # append the last observation, skipping n=SKIP_ACTIONS that were before
        new_state.append(state_)

        # update reward when losing game
        if info['lives'] < n_lifes:
            action_reward += -2

        # clip reward
        if action_reward > 1:
            action_reward = 1
        elif action_reward < -1:
            action_reward = -1
            

        score += action_reward
        frame_count = frame_count + 1
        state = new_state
        n_lifes = info['lives'] 



    scores.append(score)
    print(f'Episode {i}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}') 