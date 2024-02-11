import sys
sys.path.append("NET")
from NET.Agent import Agent
import gym
from NET.transforms import  apply_wrappers, Transforms
import numpy as np
import gc
import torch
from matplotlib import pyplot as plt
from collections import deque
  
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

# epsilon
start=1
end=0.1
final_eps = 0.01 
N_OBSERVE_STEPS = 50000 
kneepoint=1000000 
final_knee_point = 22000000


DISPLAY = False
LR = 5E-5
N_FRAMES = 4
SKIP_ACTIONS = 4

# TRAINING
UPDATE_FRAME_COUNT = 4
MAX_EPISODES = 60_000

# Initialize Gym Environment
env =gym.make( env_name, render_mode='human' if DISPLAY else 'rgb_array' )


# Create an agent
agent = Agent(state_space=(4,84,84), action_space=4, model_name='32x64x64_breakout_model', gamma=.99,
               batch_size=32, lr=LR)

agent.assign_eps(start, end, final_eps, N_OBSERVE_STEPS, kneepoint, final_knee_point)

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

    # Reset environment and preprocess state
    obs, _  = env.reset()
    state = Transforms.to_gray(obs[0])

    black_screen = np.zeros_like(state[0])
    state = deque( [state[0]]*N_FRAMES, maxlen=N_FRAMES )

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
        obs_stack.append(obs[0])

        for k in range(SKIP_ACTIONS):
            obs_, action_reward, done, trunc, info = env.step(action)
            reward += action_reward 
            obs_stack.append(obs_)
            if done: break

        next_state = state.copy()

        # if dead, reset the history, since previous states don't matter anymore
        if done: 
             next_state.append(black_screen)
        else:
            # blur last two frames
            next_obs = np.maximum(*list(obs_stack))
            # transform the observation
            state_ = Transforms.to_gray(next_obs).squeeze(0)
            # append the last observation, skipping n=SKIP_ACTIONS that were before
            next_state.append(state_)

        # debug_imgs(state, next_state)
        # exit()
            
        # update reward when losing game
        if info['lives'] < n_lifes:
            reward += -2

        # clip reward
        reward = np.sign(reward)
            
        # Preprocess next state and store transition
        agent.store_experience(state, next_state, action, reward, int(done)) 
        

        # train agent
        if (agent.step_counter > N_OBSERVE_STEPS) and (frame_count % UPDATE_FRAME_COUNT == 0):
            agent.learn()

            
        # store the scores
        score += reward
        frame_count = frame_count + 1
        state = next_state  
        n_lifes = info['lives'] 
        
    agent.update_num_episodes()

    # Maintain record of the max score achieved so far
    if score > max_score:
        max_score = score

    mean_score = np.mean(scores[-100:])

    scores.append(score)
    if(episode%20 == 0):
        print(f'Episode {episode} Step {agent.learn_counter}: \n\tScore: {score}\n\tAvg score (past 100): {mean_score}\
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