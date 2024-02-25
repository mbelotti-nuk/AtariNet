import sys
sys.path.append("NET")
from NET.Agent import Agent
import gym
from NET.transforms import Environment
import numpy as np
import gc
import torch  

# Specify environment location
env_name = "ALE/Breakout-v5"

# epsilon
EPS_START=1
EPS_END=0.1
EPS_FINAL = 0.01 
N_OBSERVE_STEPS = 50_000 
KNEE_STEP = 1_000_000 
KNEE_FINAL_STEP = 5_000_000

# specifications on environment
DISPLAY = False
LR = 5E-5
N_FRAMES = 4
SKIP_ACTIONS = 4

# TRAINING
UPDATE_FRAME_COUNT = 4
MAX_EPISODES = 100_000


# Initialize Gym Environment
env =gym.make( env_name, render_mode='human' if DISPLAY else 'rgb_array' )
ENVIRONMENT = Environment(env, skip_frames=SKIP_ACTIONS, n_frames=N_FRAMES, state_space=(4,84,84))

# Create an agent
agent = Agent(state_space=(4,84,84), action_space=4, model_name='32x64x64_breakout_model', gamma=.99,
               batch_size=32, lr=LR)
agent.define_epsilon_startegy(EPS_START, EPS_END, EPS_FINAL, N_OBSERVE_STEPS, KNEE_STEP, KNEE_FINAL_STEP)


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
    # re-set the environment
    ENVIRONMENT.reset()

    score = 0
    frame_count = 0
    

    while not done:

        # Take epsilon greedy action
        action = agent.choose_action(ENVIRONMENT.state())

        # make a step
        state, next_state, action, reward, done = ENVIRONMENT.step(action)
        
        # Preprocess next state and store transition
        agent.store_experience(state, next_state, action, reward, done) 
        
        # train agent every UPDATE_FRAME_COUNT
        if (agent.step_counter > N_OBSERVE_STEPS) and (frame_count % UPDATE_FRAME_COUNT == 0):
            agent.learn()

        # store the scores
        score += reward
        frame_count = frame_count + 1
        
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