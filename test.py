import sys
sys.path.append("NET")
from NET.Agent import Agent
import gym
from NET.transforms import Transforms
import numpy as np
import gc
import torch

# Specify environment location
env_name = 'ALE/Breakout-v5'
NUM_EPISODES = 100 
DISPLAY = True
N_FRAMES = 4


# Initialize Gym Environment
env =gym.make( 'ALE/Breakout-v5',render_mode='human' if DISPLAY else 'rgb_array' ) #render_mode="rgb_array")

# Create an agent
agent = Agent(replace_target_cnt=1000, state_space=(4,84,84), action_space=4, model_name='breakout_model', gamma=.9,
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
    state, _  = env.reset()
    state = Transforms.to_gray(state[0])
    state = np.stack([state[0]]*N_FRAMES)

    n_lifes = 5
    score = 0
            
    while not done:
        # Take epsilon greedy action
        action = agent.choose_action(state, train=False)

        # apply action for N_FRAMES consecutive frames
        action_reward = 0
        new_state = []
        new_observations = []

        for f in range(N_FRAMES):
            obs_, reward, done, trunc, info = env.step(action)
            state_ = Transforms.to_gray(obs_)
            action_reward += reward
            new_state.append(state_)
            new_observations.append(obs_)
            if done:
                for f_i in range(f+1,N_FRAMES):
                    new_state.append(np.copy(state_))
                    new_observations.append(np.copy(obs_))
                break 
        new_state = np.stack(new_state,axis=1).squeeze(0)

        # update reward when losing game
        if info['lives'] < n_lifes:
            action_reward += -2

        # clip reward
        if action_reward > 1:
            action_reward = 1
        elif action_reward < -1:
            action_reward = -1
            
        score += action_reward
        state = new_state
        n_lifes = info['lives'] 


    scores.append(score)
    print(f'Episode {i}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}') 