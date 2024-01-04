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
NUM_EPISODES = 40_000 
DISPLAY = False
LR = 1E-4
N_FRAMES = 4
EPS_STRT = 1
EPS_MIN = .1
EPS_DEC = 0.99992
MIN_EPISODES_TO_LEARN = 2
UPDATE_FRAME_COUNT = 4

# Initialize Gym Environment
env =gym.make( 'ALE/Breakout-v5',render_mode='human' if DISPLAY else 'rgb_array' )

# Create an agent
agent = Agent(state_space=(4,84,84), action_space=4, model_name='32x64x64_breakout_model', gamma=.9,
                eps_strt=EPS_STRT, eps_min=EPS_MIN , eps_dec=EPS_DEC, batch_size=32, lr=LR, number_frames=N_FRAMES)


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
    frame_count = 0
            
    while not done:
        # Take epsilon greedy action
        action = agent.choose_action(state)

        # apply action for N_FRAMES consecutive frames
        action_reward = 0
        new_state = []
        new_observations = []

        # take N_FRAMES consecutive actions
        for f in range(N_FRAMES):
            obs_, reward, done, trunc, info = env.step(action)
            state_ = Transforms.to_gray(obs_)
            action_reward += reward
            new_state.append(state_)
            #new_observations.append(obs_)
            if done:
                for f_i in range(f+1,N_FRAMES):
                    new_state.append(np.copy(state_))
                    #new_observations.append(np.copy(obs_))
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
            

        # Preprocess next state and store transition
        agent.store_transition(state, action, action_reward, new_state, int(done), new_observations) #np.stack(new_observations,axis=1))

        # train agent
        if (i > MIN_EPISODES_TO_LEARN) and (frame_count % UPDATE_FRAME_COUNT == 0):
            agent.learn(num_game=i)

        score += action_reward
        frame_count = frame_count + 1 if frame_count <= UPDATE_FRAME_COUNT else 0
        state = new_state
        n_lifes = info['lives'] 

    # decrease epsilon greedy
    agent.dec_eps()
    # Maintain record of the max score achieved so far
    if score > max_score:
        max_score = score

    # # Save a gif if episode is best so far
    # if score > 50 and score >= max_score:
    #      agent.save_gif(cnt)

    scores.append(score)
    print(f'Episode {i}: \n\tScore: {score}\n\tAvg score (past 100): {np.mean(scores[-100:])}\
                \n\tEpsilon: {agent.eps}\n')
            

print("save model")
agent.policy_net.save_model()
agent.plot_results(scores)
torch.cuda.empty_cache()
env.close()