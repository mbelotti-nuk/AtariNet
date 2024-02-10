import sys
sys.path.append("NET")
from NET.Agent import Agent
import gym
from NET.transforms import  apply_wrappers
import numpy as np
import gc
import torch
from matplotlib import pyplot as plt
import random 
from collections import deque

def reset_env():
    global env
    global NOOP_MAX
    obs, _  = env.reset()
    #for _ in range(random.randint(1, NOOP_MAX)):
    obs, _, _, _, _ = env.step(1)
    return obs   

def debug_imgs(state, next_state): 
    for i in range(0, len(next_state)):        
        plt.imshow(next_state[i], cmap="gray")
        plt.savefig(f"next_state{i}.png")
        plt.clf()
        plt.imshow(state[i], cmap="gray")
        plt.savefig(f"state{i}.png")
        plt.clf()
    return

# Specify environment location
env_name = "ALE/Breakout-v5"
OBSERVE = 50_000 # (steps)
EXPLORE = 5_000_000 # (steps)
MAX_TRAIN = 5_500_000 #(steps)

DISPLAY = False
LR = 0.00025
N_FRAMES = 4
SKIP_ACTIONS = 4
NOOP_MAX = 30

EPS_STRT = 1
EPS_MIN = .1
EPS_DEC = (EPS_STRT -EPS_MIN)  / EXPLORE
print(f"Epsilon decay {EPS_DEC}")

# TRAINING
UPDATE_FRAME_COUNT = 4
MAX_EPISODES = 80_000

# Initialize Gym Environment
env =gym.make( env_name, render_mode='human' if DISPLAY else 'rgb_array' )

# apply wrappers
env = apply_wrappers(env=env)

# define no-op action
noop_action = 0
assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

# Create an agent
agent = Agent(state_space=(4,84,84), action_space=4, model_name='32x64x64_breakout_model', gamma=.99,
                eps_strt=EPS_STRT, eps_min=EPS_MIN , eps_dec=EPS_DEC, batch_size=32, lr=LR, number_frames=N_FRAMES)


# Clean environment
if torch.cuda.is_available():
    # Empty cache in cuda if needed
    torch.cuda.empty_cache()
    # Garbage collect
    gc.collect()

scores = []
max_score = 0



for episode in range(0, MAX_EPISODES):
    done = False

    if(agent.step_counter>MAX_TRAIN):
        break
         

    # Reset environment and preprocess state
    #state, _  = env.reset()
    obs_ = reset_env()
    black_screen = np.zeros_like(obs_)
    state = deque( [obs_]*N_FRAMES, maxlen=N_FRAMES )
    score = 0
    frame_count = 0
    n_lifes = 5

    while not done:

        # Take epsilon greedy action
        action = agent.choose_action(state)
        reward = 0

        # take next actions: the number of consecutive actions is qeual to SKIP_ACTIONS
        # check: https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
        obs_stack = deque(maxlen=2)
        obs_stack.append(obs_[0])
        for k in range(SKIP_ACTIONS):
            obs_, action_reward, done, trunc, info = env.step(action)
            reward += action_reward 
            obs_stack.append(obs_)
            if done: break

        next_state = state.copy()

        # if dead, reset the history, since previous states don't matter anymore
        if done: #info['lives'] < n_lifes:
             next_state.append(black_screen)
        else:
            # blur last two frames
            next_obs = np.maximum(*list(obs_stack))
            # append the last observation, skipping n=SKIP_ACTIONS that were before
            next_state.append(next_obs )

        # debug_imgs(state, next_state)
        # exit()

        n_lifes = info['lives'] 

        # clip reward
        reward = np.sign(reward)
            
        # Preprocess next state and store transition
        agent.store_transition(state, action, reward, next_state, int(done)) 

        # train agent
        if (agent.step_counter > OBSERVE) and (frame_count % UPDATE_FRAME_COUNT == 0):
            agent.learn()

        score += reward
        frame_count = frame_count + 1
        state = next_state  
        
        if (agent.step_counter > OBSERVE):
            agent.dec_eps()
        
        agent.update_counter()


    # Maintain record of the max score achieved so far
    if score > max_score:
        max_score = score

    mean_score = np.mean(scores[-100:])

    scores.append(score)
    if(episode%10 == 0):
        print(f'Episode {episode} Step {agent.step_counter}: \n\tScore: {score}\n\tAvg score (past 100): {mean_score}\
                \n\tEpsilon: {agent.eps}\n')

    if mean_score > 30:
        agent.policy_net.save_model()
        print("FINISHED")
        break
            

print("save model")
agent.policy_net.save_model()
agent.plot_results(scores)
torch.cuda.empty_cache()
env.close()