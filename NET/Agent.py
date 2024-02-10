from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy as np
import random
from replay_mem import ReplayBuffer
from NeuralNet import Dueling_DQNnet, DQNnet
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import os

# This class trains and plays on the actual game
class Agent(object):
    def __init__(self, state_space, action_space, 
                model_name='breakout_model', number_frames = 4, gamma=0.99, eps_strt=1, 
                eps_min=0.1, eps_dec=5e-6, batch_size=32, lr=0.001):

        # number of frames to be fed to the network
        self.number_frames = number_frames
        # state space of the game: it gets the screen of the game
        self.state_space = state_space
        # represent the possible input that can be done while playing
        self.action_space = action_space
        # batch size for training the network
        self.batch_size = batch_size
         # After how many training iterations the target network should update
        self.sync_network_rate = 20_000


        self.GAMMA = gamma
        # learning rate
        self.LR = lr
        # epsilon greedy coeff
        self.eps = eps_strt
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.step_counter = 0
        self.save_interval = 500_000

        # Use GPU if available
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"DEVICE {self.device}")

        # Initialise Replay Memory
        self.memory = ReplayBuffer()


        # Initialise policy and target networks, set target network to eval mode
        self.policy_net = DQNnet(input_dim=self.state_space, out_dim=self.action_space, filename=model_name).to(self.device)
         #Dueling_DQNnet(input_dim=self.state_space, out_dim=self.action_space, filename=model_name).to(self.device)
        print(self.policy_net)
        self.target_net = DQNnet(input_dim=self.state_space, out_dim=self.action_space, filename=model_name+'target').to(self.device)
         #Dueling_DQNnet(input_dim=self.state_space, out_dim=self.action_space, filename=model_name+'target').to(self.device)
        self.target_net.eval()

        # If pretrained model of the modelname already exists, load it
        try:
            self.policy_net.load_model()
            print('loaded pretrained model')
        except:
            pass
        
        # Set target net to be the same as policy net
        self.sync_networks()

        # Set optimizer & loss function
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.loss = torch.nn.MSELoss() #torch.nn.HuberLoss() #torch.nn.SmoothL1Loss()

    def plot_results(self, scores, save_path=None):
        plt.plot(np.arange(1, len(scores)+1), scores, label = "Scores per game", color="blue")
        plt.plot( np.convolve(scores, np.ones(100)/100, mode='valid'), label = "Moving mean scores", color="red")
        plt.title("Scores")
        plt.xlabel("Game")
        plt.legend()
        if save_path !=None:
            plt.savefig(os.path.join(save_path,"SCORES.png"))
        else:
            plt.savefig("SCORES.png")
        plt.close()    


    def sync_networks(self):
        if self.step_counter % self.sync_network_rate == 0 and self.step_counter > 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def sample_batch(self):
        batch = self.memory.sample_batch(self.batch_size)
        state_shape = batch.state[0].shape

        # Convert to tensors with correct dimensions
        state =  torch.tensor( batch.state  ).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        action = torch.tensor( batch.action ).unsqueeze(1).to(self.device)
        reward = torch.tensor( batch.reward ).float().unsqueeze(1).to(self.device)
        state_ = torch.tensor( batch.next_state ).view(self.batch_size, -1, state_shape[1], state_shape[2]).float().to(self.device)
        done =   torch.tensor( batch.done   ).float().unsqueeze(1).to(self.device)

        return state, action, reward, state_, done

    # Returns the greedy action according to the policy net
    def greedy_action(self, obs):
        obs = np.stack(list(obs), axis=0)
        obs = torch.tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        action = self.policy_net(obs).argmax().item()
        return action

    # Returns an action based on epsilon greedy method
    def choose_action(self, obs, train=True):
        if train:
            # choose action from model 
            if random.random() > self.eps:
                action = self.greedy_action(obs)
            # return random action 
            else:
                action = random.choice([x for x in range(self.action_space)])
        else:
            action = self.greedy_action(obs)
        return action
    
    # Stores a transition into memory
    def store_transition(self, *args):
        self.memory.add_transition(*args)


    # Decrement epsilon 
    def dec_eps(self):
        self.eps = max(self.eps - self.eps_dec, self.eps_min)

    # Samples a single batch according to batchsize and updates the policy net
    def learn(self):

        if len(self.memory.buffer) < self.batch_size:
            return 

        # Sample batch
        # "state", "action", "reward", "next_state", "done"
        state, action, reward, state_, done = self.sample_batch()

        self.optim.zero_grad()
        # Calculate the value of the action taken
        q_eval = self.policy_net(state).gather(1, action)
        
        # Using q_next and reward, calculate q_target
        # (1-done) ensures q_target is 0 if transition is in a terminating state
        with torch.no_grad():
            # Calculate best next action value from the target net and detach from grap
            q_next = self.target_net(state_).detach().max(1)[0].unsqueeze(1)
        
        TD_error = (1-done) * (reward + self.GAMMA * q_next) + (done * reward)

        # Compute the loss
        loss = self.loss(q_eval, TD_error).to(self.device)

        # Perform backward propagation and optimization step
        loss.backward()
        self.optim.step()

        self.sync_networks()

        # Save model & decrement epsilon
        if (self.step_counter) % self.save_interval == 0:
            self.policy_net.save_model()


    def update_counter(self):
        self.step_counter += 1
        return

    # Save gif of an episode starting num_transitions ago from memory
    def save_gif(self, frames_raw):
        frames = [Image.fromarray(image, mode='RGB')for image in frames_raw]
        frame_one = frames[0]
        frame_one.save("match.gif", format="GIF", append_images=frames,
                save_all=True, duration=100, loop=0)
        