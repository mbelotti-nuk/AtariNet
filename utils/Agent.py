from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import numpy as np
import random
from replay_mem import  experience, replay_buffer, PER_replay_buffer
from NeuralNet import Dueling_DQNnet, DQNnet
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# This class trains and plays on the actual game
class Agent(object):
    def __init__(self, state_space, action_space, 
                model_name='breakout_model', gamma=0.99,
                batch_size=32, lr=0.001,
                prioritized_replay=False,
                dueling=False):

        self.state_space = state_space
        self.action_space = action_space
        # batch size for training the network
        self.batch_size = batch_size
         # After how many training iterations the target network should update
        self.sync_network_rate = 10_000


        self.GAMMA = gamma
        self.TAU = 1e-3              # Soft update parameter for updating fixed q network
        # learning rate
        self.LR = lr
        
        self.learn_counter = 0
        self.step_counter = 0
        self.num_episodes = 0

        # Use GPU if available
        if torch.cuda.is_available():
            type_dev = "cuda:0"
        elif torch.backends.mps.is_available():
            type_dev = "mps:0"
        else: 
            type_dev = "cpu"
        
        self.device = torch.device(type_dev)

        self.prioritized_replay = prioritized_replay
        
        # Initialize Replay Memory
        if(self.prioritized_replay):
            self.memory = PER_replay_buffer(alfa=0.6) 
        else:
            self.memory = replay_buffer() 

        self.dueling = dueling
        # Initialise policy and target networks, set target network to eval mode
        if dueling:
            self.policy_net = Dueling_DQNnet(input_dim=state_space, out_dim=action_space, filename=model_name).to(self.device)
            self.target_net = Dueling_DQNnet(input_dim=state_space, out_dim=action_space, filename=model_name+'target').to(self.device)
        else:
            self.policy_net = DQNnet(input_dim=state_space, out_dim=action_space, filename=model_name).to(self.device)
            self.target_net = DQNnet(input_dim=state_space, out_dim=action_space, filename=model_name+'target').to(self.device)
        self.target_net.eval()


        self.policy_net._initialize_weights()

        
        # Set target net to be the same as policy net
        self.sync_networks()

        # Set optimizer & loss function
        self.optim = torch.optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.loss = torch.nn.HuberLoss() 


    def define_epsilon_startegy(self, eps_start, eps_end, final_eps, 
                                step_start, step_knee, step_final_knee):
        self.eps = 1
        # epsilon greedy startegy
        self.eps_strategy = EpsilonScheduler(eps_start = eps_start, eps_end = eps_end, final_eps = final_eps , 
                                            step_start = step_start, kneepoint = step_knee, final_knee_point = step_final_knee)


    def load_net(self, path):
        self.policy_net.load_model(path=path)

    @property
    def frame_counter(self):
        return self.step_counter

    # Copies policy params to target net
    def sync_networks(self, soft=False):
        if soft:
            for source_parameters, target_parameters in zip(self.policy_net.parameters(), self.target_net.parameters()):
                target_parameters.data.copy_((1.0 - self.TAU)* source_parameters.data + self.TAU * target_parameters.data)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def update_num_episodes(self):
        self.num_episodes += 1
        if(self.prioritized_replay):
            self.memory.beta_annealing_schedule


    # Returns the greedy action according to the policy net
    def greedy_action(self, obs):
        obs = np.stack(list(obs), axis=0)
        obs = torch.tensor(obs).float().to(self.device)
        obs = obs.unsqueeze(0)
        with torch.no_grad():
            action = self.policy_net(obs).argmax().item()
        return action

    # Returns an action based on epsilon greedy method
    def choose_action(self, obs, train=True):
        if train:
            self.step_counter += 1
            self.eps = self.eps_strategy.get_exploration_rate(self.step_counter)
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
    def store_experience(self, *args):
        self.memory.add_experience(experience(*args))

    # Samples a single batch according to batchsize and updates the policy net
    def learn(self):

        if self.memory.buffer_length < self.batch_size:
            return 

        # Sample batch
        experiences = self.memory.sample_batch(batch_size=self.batch_size, device=self.device)
        
        q_eval, q_target = self.agent_predictions(experiences)

        if self.prioritized_replay:
            TD_errors = torch.abs(q_eval - q_target)
            # update memory 
            self.memory.update_priorities( experiences.index , TD_errors.detach().cpu().numpy().flatten())  
            # compute loss
            sampling_weights = (torch.Tensor(experiences.weight).view(-1,self.batch_size)).to(self.device)
            loss = torch.mean((TD_errors * sampling_weights)**2)
                     
        else:
            # Compute the loss
            loss = self.loss(q_eval, q_target).to(self.device)


        # Perform backward propagation and optimization step
        loss.backward()
        # clip gradient norm
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optim.step()
        self.optim.zero_grad()

        self.learn_counter += 1

        # sync target and policy networks
        # the update of the target is made with a frequency w.r.t. the number of steps done
        # and not w.r.t. the number of parameters (==self.learn_counter)
        if self.step_counter % self.sync_network_rate == 0 and self.learn_counter > 0:
            self.sync_networks()

        # Save model
        # if (self.learn_counter) % self.save_interval == 0:
        #     self.policy_net.save_model()
    
    def agent_predictions(self, experience):

        if self.dueling:
            q_eval = self.policy_net(experience.state)
            q_next = self.target_net(experience.next_state)
            # choose action with policy network
            action = torch.argmax(q_eval, dim=1, keepdim=True)
            # evaluate q_next with the action selected by policy network
            q_target = (1-experience.done) * (experience.reward + self.GAMMA * q_next.gather(1, action).detach()) + (experience.done * experience.reward)
            q_eval = q_eval.gather(1,experience.action)
        else:
            q_eval = self.policy_net(experience.state).gather(1, experience.action)
            q_next = self.target_net(experience.next_state).detach().max(1)[0].unsqueeze(1)
            q_target = (1-experience.done) * (experience.reward + self.GAMMA * q_next) + (experience.done * experience.reward)

        return q_eval, q_target

        
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


class EpsilonScheduler():
    def __init__(self,  eps_start,  eps_end, final_eps = None, 
                        step_start = 50_000, kneepoint=1_000_000, final_knee_point = 5_000_000):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.final_eps = final_eps
        self.kneepoint = kneepoint
        self.startpoint = step_start
        self.final_knee_point = final_knee_point

    def get_exploration_rate(self, current_step):
        if current_step < self.startpoint:
            return 1.
        else:
            if current_step < self.kneepoint:
                return self.eps_start - (current_step-self.startpoint) * (self.eps_start-self.eps_end)/(self.kneepoint-self.startpoint)
            else:
                if self.final_eps is None:
                    return self.eps_end
                else:
                    epsilon = self.eps_end - (current_step-self.kneepoint) * (self.eps_end-self.final_eps)/(self.final_knee_point-self.kneepoint)
                    return max(epsilon, self.final_eps)