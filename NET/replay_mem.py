import random
from collections import namedtuple
import numpy as np
import torch
from utils import SumTree
from typing import List

transition = namedtuple('transition', ("state", "action", "reward", "done"))
experience = namedtuple('experience', ("state", "next_state", "action", "reward", "done"))
per_experience = namedtuple('experience', ("state", "next_state", "action", "reward", "done", "index", "weight"))



class PER_replay_buffer(object):  
    error_epsilon = 0.01  

    def __init__(self, alfa, capacity=200_000):
        # Making the tree 
        self.tree = SumTree(capacity)
        self._alfa = alfa
        self._beta = 0.4
        self._beta_increment_per_sampling = 0.00001
        self.dtype = np.uint8
        self.max_priority = 1

    @property
    def buffer_length(self):
        return self.tree.n_entries

    def add_experience(self, _experience:experience):
        # transform to unit8
        _state = np.array(_experience.state*255, dtype = self.dtype)
        # save only last frame
        _next_state = np.array(_experience.next_state[-1]*255, dtype=self.dtype)
        # make torch tensors
        new_experience = experience(state =  _state, 
                                    next_state = _next_state,
                                    action = _experience.action, 
                                    reward = _experience.reward, 
                                    done =   _experience.done)


        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if self.max_priority == 0:
            self.max_priority = self.absolute_error_upper

        self.tree.add(self.max_priority, new_experience)   # set the max priority for new priority


    def sample_batch(self, batch_size, device="cuda:0"):
        # Create a minibatch array that will contains the minibatch
        minibatch = []
        priorities = []
        batch_idx = np.empty((batch_size,), dtype=np.int32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / batch_size       # priority segment

        for i in range(batch_size):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)
            
            priorities.append(priority)
            batch_idx[i]= index

            # re-construct next_state
            _next_state = torch.cat(( data[0][1:,:,:], data[1].unsqueeze(0) ), dim=0)
            minibatch.append(experience(  state = data[0].unsqueeze(0).float().cuda()/255, 
                                          next_state = _next_state.unsqueeze(0).float().cuda()/255, 
                                          action =  data[2].reshape(1).cuda(),
                                          reward =  data[3].reshape(1).cuda(),
                                          done =    data[4].reshape(1).cuda()))

        # compute weight
        possibilities = priorities / self.tree.total_priority
        weight = np.power(self.tree.n_entries * possibilities, -self._beta)
        max_weight = weight.max()
        weight = weight/max_weight
        weight = torch.tensor(weight[:,np.newaxis], dtype = torch.float).to(device)

        per_experiences = self._extract_tensors(experiences=minibatch, experience_index=batch_idx,weight=weight )
        return per_experiences
    

    def update_priorities(self, tree_idx, abs_errors):
        priorities_list = (np.abs(abs_errors) + self.error_epsilon) ** self._alfa
        for index, priority in zip(tree_idx, priorities_list):
            self.tree.update(idx=index, priority=priority)
            self.max_priority = max(self.max_priority, priority)

    @property
    def beta_annealing_schedule(self):
        self._beta = min(1.0, self._beta + self._beta_increment_per_sampling )

    def _extract_tensors(self, experiences:List[experience], experience_index, weight) -> per_experience:
        # Convert batch of Experiences to Experience of batches
        batch = experience(*zip(*experiences))

        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.next_state)
        t3 = torch.cat(batch.action).squeeze(0).unsqueeze(1)
        t4 = torch.cat(batch.reward).squeeze(0).unsqueeze(1)
        t5 = torch.cat(batch.done).squeeze(0).unsqueeze(1)

        return per_experience(t1,t2,t3,t4,t5, experience_index, weight)


# Memory which allows for storing and sampling batches of transitions
class replay_buffer(object):
    def __init__(self):
        self.replay_memory_size = 1_000_000
        self.buffer = np.empty(self.replay_memory_size, dtype = [("transition", transition)] )
        self.pointer = 0
        self.dtype = np.uint8

    # Adds a single experience to the memory buffer
    def add_experience(self, current_experience): 
        
        states = np.array(current_experience.state[-1]*255, dtype = self.dtype) 
        current_transition = transition(state=states, action=current_experience.action, reward=current_experience.reward, done=current_experience.done)
        
        if(self.pointer < self.replay_memory_size):
            self.buffer[self.pointer]["transition"] = current_transition
        else:
            self.buffer[self.pointer % self.replay_memory_size]["transition"] = current_transition

        self.pointer += 1

    @property
    def buffer_length(self):
        if(self.pointer < self.replay_memory_size):
            return self.pointer
        else:
            return self.replay_memory_size 
    @property
    def buffer_end(self):
        if(self.pointer < self.replay_memory_size):
            return self.pointer -1
        else:
            return self.replay_memory_size -1

    # Samples a batch of transitions
    def sample_batch(self, batch_size=64, device="cuda:0"):

        begin = 3
        end = self.buffer_end
        
        batch_index = np.random.randint(begin, end, size = batch_size)
        
        states, next_states, actions, rewards, dones = self.to_arrays(batch_index)

        state_shape = states[0].shape

        # Convert to tensors with correct dimensions
        state =  (torch.tensor( states  ).view(batch_size, -1, state_shape[1], state_shape[2]).float()/255).to(device)
        action = torch.tensor( actions ).unsqueeze(1).type(torch.int64).to(device)
        reward = torch.tensor( rewards ).float().unsqueeze(1).to(device)
        next_state = (torch.tensor( next_states ).view(batch_size, -1, state_shape[1], state_shape[2]).float()/255).to(device)
        done =   torch.tensor( dones   ).float().unsqueeze(1).to(device)


        return experience(state, next_state, action, reward, done)
    
    def to_arrays(self, batch_index):
        states, next_states = [], []
        for ind in batch_index:
            
            state = np.stack( [self.buffer["transition"][ind+i].state for i in range(-3,1)])
            next_state = np.stack( [self.buffer["transition"][ind+i].state for i in range(-2,2)])

            states.append(state)
            next_states.append(next_state)
        
        states = np.stack( states ) 
        next_states = np.stack(next_states ) 

        transitions = self.buffer["transition"][batch_index]
        actions = np.stack( [exp.action for exp in transitions] ) 
        rewards = np.stack( [exp.reward for exp in transitions] ) 
        dones = np.stack( [exp.done for exp in transitions] ) 


        return states, next_states, actions, rewards, dones
