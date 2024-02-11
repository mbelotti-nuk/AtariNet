import random
from collections import namedtuple, deque
import numpy as np
import torch

experience = namedtuple('experience', ("state", "next_state", "action", "reward", "done"))
per_experience = namedtuple('experience', ("state", "next_state", "action", "reward", "done", "index", "weight"))

# Memory which allows for storing and sampling batches of transitions
class ReplayBuffer(object):
    def __init__(self):
        self.replay_memory_size = 1_000_000
        self.buffer = np.empty(self.replay_memory_size, dtype = [("experience", experience)] )
        self.pointer = 0
        self.dtype = np.uint8

    # Adds a single transitions to the memory buffer
    def add_experience(self, current_experience): 

        if(self.pointer < self.replay_memory_size):
            self.buffer[self.pointer]["experience"] = current_experience
        else:
            self.buffer[self.pointer % self.replay_memory_size]["experience"] = current_experience

        self.pointer += 1

    def buffer_length(self):
        if(self.pointer < self.replay_memory_size):
            return self.pointer
        else:
            return self.replay_memory_size

    # Samples a batch of transitions
    def sample_batch(self, batch_size=64, device="cuda:0"):
        batch_index = np.random.randint(0, self.buffer_length(), size = batch_size)
        experiences = self.buffer["experience"][batch_index]
        
        states, next_states, actions, rewards, dones = self.to_arrays(experiences)

        state_shape = states[0].shape

        # Convert to tensors with correct dimensions
        state =  torch.tensor( states  ).view(batch_size, -1, state_shape[1], state_shape[2]).float().to(device)
        action = torch.tensor( actions ).unsqueeze(1).type(torch.int64).to(device)
        reward = torch.tensor( rewards ).float().unsqueeze(1).to(device)
        next_state = torch.tensor( next_states ).view(batch_size, -1, state_shape[1], state_shape[2]).float().to(device)
        done =   torch.tensor( dones   ).float().unsqueeze(1).to(device)

        return experience(state, next_state, action, reward, done)
    

    def to_arrays(self, experiences):
        states = np.array( [exp.state for exp in experiences] ) 
        next_states = np.array( [exp.next_state for exp in experiences] ) 
        actions = np.array( [exp.action for exp in experiences] ) 
        rewards = np.array( [exp.reward for exp in experiences] ) 
        dones = np.array( [exp.done for exp in experiences] ) 
        return states, next_states, actions, rewards, dones


class PrioritizedReplayBuffer(object):
        def __init__(self, alfa):
            self.replay_memory_size = 1_000_000
            self.buffer = np.empty(self.replay_memory_size, dtype = [("experience", experience)] )
            self.priorities = np.empty(self.replay_memory_size, dtype=np.float32)
            self.pointer = 0
            self.dtype = np.uint8
            self._alfa = alfa
            self._beta = 1
            self.random_state = np.random.RandomState()

        # Adds a single transitions to the memory buffer
        def add_experience(self, current_experience):
            priority = 1
            if(self.pointer > 1 ):
                priority = np.max(self.priorities) 

            if(self.pointer < self.replay_memory_size):
                self.buffer[self.pointer]["experience"] = current_experience
                self.priorities[self.pointer] = priority
            else:
                # subsitute lowest priority
                if priority > self.priorities.min():
                    index = self.priorities.argmin()
                    self.buffer[index]["experience"] = current_experience
                    self.priorities[index] = priority
                # do not add low priorities
                else:
                    pass

            self.pointer += 1


        def buffer_length(self):
            if(self.pointer < self.replay_memory_size):
                return self.pointer
            else:
                return self.replay_memory_size


        def sample_batch(self, batch_size=64, device="cuda:0"):
            # use sampling scheme to determine which experiences to use for learning
            ps = self.priorities[ : self.buffer_length() ]
            sampling_probs = ps**self._alfa / np.sum(ps**self._alfa)
            batch_index = self.random_state.choice(np.arange(self.buffer_length()),
                                            size=batch_size,
                                            replace=True,
                                            p=sampling_probs)
            

            experiences = self.buffer["experience"][batch_index]
            
            states, next_states, actions, rewards, dones = self.to_arrays(experiences)

            state_shape = states[0].shape

            # Convert to tensors with correct dimensions
            state =  torch.tensor( states  ).view(batch_size, -1, state_shape[1], state_shape[2]).float().to(device)
            action = torch.tensor( actions ).unsqueeze(1).type(torch.int64).to(device)
            reward = torch.tensor( rewards ).float().unsqueeze(1).to(device)
            next_state = torch.tensor( next_states ).view(batch_size, -1, state_shape[1], state_shape[2]).float().to(device)
            done =   torch.tensor( dones   ).float().unsqueeze(1).to(device)

                
            weights = (self.buffer_length() * sampling_probs[batch_index])**-self._beta
            normalized_weights = weights / weights.max()
            
            return per_experience(state, next_state, action, reward, done, batch_index, normalized_weights)

        def beta_annealing_schedule(self, num_episodes):
            self._beta =  1 - np.exp(-1e-3 * num_episodes)

        def update_priorities(self, indices: np.array, priorities: np.array):
            self.priorities[indices] = priorities
        
        def to_arrays(self, experiences):
            states = np.array( [exp.state for exp in experiences] ) 
            next_states = np.array( [exp.next_state for exp in experiences] ) 
            actions = np.array( [exp.action for exp in experiences] ) 
            rewards = np.array( [exp.reward for exp in experiences] ) 
            dones = np.array( [exp.done for exp in experiences] ) 
            return states, next_states, actions, rewards, dones