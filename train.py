import sys
sys.path.append("NET")
from utils.Agent import Agent
import gym
from utils.game_environment import Environment
import numpy as np
import gc
import torch  

# Specify environment location
env_name = "BreakoutNoFrameskip-v4"

# epsilon
EPS_START=1
EPS_END = 0.1
EPS_FINAL = None #0.05 
N_OBSERVE_STEPS = 50_000 
KNEE_STEP = 5_000_000 
KNEE_FINAL_STEP = 10_000_000

# specifications on environment
DISPLAY = False
LR = 2.5E-5
N_FRAMES = 4
SKIP_ACTIONS = 4

# TRAINING
UPDATE_FRAME_COUNT = 4
MAX_EPISODES = 30_000
MAX_FRAMES = 20_000_000
MAX_SCORE = 140


# Initialize Gym Environment
env =gym.make( env_name, render_mode='human' if DISPLAY else 'rgb_array' )
ENVIRONMENT = Environment(env, skip_frames=SKIP_ACTIONS, n_frames=N_FRAMES, state_space=(4,84,84))

# Create an agent
agent = Agent(state_space=(4,84,84), action_space=4, model_name='32x64x64_breakout_model', gamma=.99,
               batch_size=32, lr=LR, dueling=False)
agent.define_epsilon_startegy(eps_start = EPS_START, eps_end = EPS_END, final_eps = EPS_FINAL, 
                              step_start = N_OBSERVE_STEPS, step_knee = KNEE_STEP, step_final_knee = KNEE_FINAL_STEP)


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
    

    while not ENVIRONMENT.end_game:

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


    if(episode % 10 == 0):
        print(f'Episode {episode} Step {agent.learn_counter}: \n\tScore: {score}\n\tAvg score (past 100): {mean_score}\
                \n\tEpsilon: {agent.eps}\n')

    if agent.frame_counter > MAX_FRAMES or mean_score > MAX_SCORE:
        break
            

print("save model")
agent.policy_net.save_model()
agent.plot_results(scores)
torch.cuda.empty_cache()
env.close()