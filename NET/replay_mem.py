import random
from collections import namedtuple, deque
import numpy as np
import torch
from utils import SumTree
from typing import List

transition = namedtuple('transition', ("state", "action", "reward", "done"))
experience = namedtuple('experience', ("state", "next_state", "action", "reward", "done"))
per_experience = namedtuple('experience', ("state", "next_state", "action", "reward", "done", "index", "weight"))


class PER_memory_buffer(object):  # stored as ( state, action, reward, next_state ) in SumTree
    error_epsilon = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, alfa, capacity=500_000):
        # Making the tree 
        self.tree = SumTree(capacity)
        self._alfa = alfa
        self._beta = 0.4
        self._beta_increment_per_sampling = 0.000005
        self.dtype = np.uint8

    @property
    def buffer_length(self):
        return self.tree.n_entries

    def add_experience(self, _experience:experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # transform to unit8
        _state = np.array(_experience.state*255, dtype = self.dtype)
        _next_state = np.array(_experience.next_state*255, dtype=self.dtype)
        # make torch tensors
        new_experience = experience(state = torch.tensor(_state), 
                                    next_state = torch.tensor(_next_state),
                                    action = torch.tensor(_experience.action), 
                                    reward = torch.tensor(_experience.reward), 
                                    done = torch.tensor(_experience.done))


        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, new_experience)   # set the max priority for new priority


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
            minibatch.append(experience(  state = data[0].unsqueeze(0).float().cuda()/255, 
                                          next_state = data[1].unsqueeze(0).float().cuda()/255, 
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

    def beta_annealing_schedule(self, num_episodes):
        self._beta =  1 - np.exp(-1e-3 * num_episodes)

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
class ReplayBuffer_memory(object):
    def __init__(self):
        self.replay_memory_size = 1_000_000
        self.buffer = np.empty(self.replay_memory_size, dtype = [("transition", transition)] )
        self.pointer = 0
        self.dtype = np.uint8

    # Adds a single experience to the memory buffer
    def add_experience(self, current_experience): 
        
        states = np.array(current_experience.state*255, dtype = self.dtype) 
        current_transition = transition(state=states, action=current_experience.action, reward=current_experience.reward, done=current_experience.done)
        
        if(self.pointer < self.replay_memory_size):
            self.buffer[self.pointer]["transition"] = current_transition
        else:
            self.buffer[self.pointer % self.replay_memory_size]["transition"] = current_transition

        self.pointer += 1

    def buffer_end(self):
        if(self.pointer < self.replay_memory_size):
            return self.pointer -1
        else:
            return self.replay_memory_size -1

    # Samples a batch of transitions
    def sample_batch(self, batch_size=64, device="cuda:0"):

        begin = 3
        end = self.buffer_end()
        
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




class ReplayBuffer_memory(object):

    # Memory replay with priorited experience replay
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, capacity=500_000, alfa=0.6, beta_start=0.4, beta_startpoint=50000, beta_kneepoint = 1000000, error_epsilon=1e-5):
        self.capacity = capacity
        self.buffer = []
        self.priority_tree = SumTree(self.capacity) # store priorities
        self.alpha = alfa
        self.beta = beta_start
        self.beta_increase = 1/(beta_kneepoint - beta_startpoint)
        self.error_epsilon = error_epsilon
        self.push_count = 0
        self.dtype = np.uint8
        self.max_priority = 1

    @property
    def buffer_length(self):
        return len(self.buffer)
    def add_experience(self, _experience:experience):
        _state = np.array(_experience.state*255, dtype = self.dtype)
        _next_state = np.array(_experience.next_state*255, dtype=self.dtype)
        new_experience = experience(state = torch.tensor(_state), 
                                    next_state = torch.tensor(_next_state),
                                    action = torch.tensor(_experience.action), 
                                    reward = torch.tensor(_experience.reward), 
                                    done = torch.tensor(_experience.done))

        if len(self.buffer) < self.capacity:
            self.buffer.append(new_experience)
        else:
            self.buffer[self.push_count % self.capacity] = new_experience

        self.push_count += 1
        # push new state to priority tree
        self.priority_tree.add(priority=1)

    def sample_batch(self, batch_size, device):
        # get indices of experience by priorities
        experience_index = []
        experiences = []
        priorities = []
        segment = self.priority_tree.total()/batch_size
        self.beta = np.min([1., self.beta + self.beta_increase])
        for i in range(batch_size):
            
            low = segment * i
            high = segment * (i+1)

            s = random.uniform(low, high)
            index, p = self.priority_tree.get(s)

            experience_index.append(index)
            priorities.append(p)

            if self.push_count > self.capacity:
                index = np.max(index,0)
            print(f"sampled index {index}, current length {len(self.buffer)}")
            experiences.append(experience(state = self.buffer[index].state.unsqueeze(0).float().cuda()/255, 
                                          next_state = self.buffer[index].next_state.unsqueeze(0).float().cuda()/255, 
                                          action =  self.buffer[index].action.reshape(1).cuda(),
                                          reward =  self.buffer[index].reward.reshape(1).cuda(),
                                          done =    self.buffer[index].done.reshape(1).cuda()))
        # compute weight
        possibilities = priorities / self.priority_tree.total()
        weight = np.power(self.priority_tree.n_entries * possibilities, -self.beta)
        max_weight = weight.max()
        weight = weight/max_weight
        weight = torch.tensor(weight[:,np.newaxis], dtype = torch.float).to(device)
 
        per_experiences = self.extract_tensors(experiences, experience_index, weight)
        return per_experiences

    def update_priorities(self, index_list, TD_error_list):
        # update priorities from TD error
        # priorities_list = np.abs(TD_error_list) + self.error_epsilon
        priorities_list = (np.abs(TD_error_list) + self.error_epsilon) ** self.alpha
        for index, priority in zip(index_list, priorities_list):
            self.priority_tree.update(idx=index, priority=priority)
            self.max_priority = np.max(self.max_priority, priority)

    def can_provide_sample(self, batch_size, replay_start_size):
        return (len(self.buffer) >= replay_start_size) and (len(self.buffer) >= batch_size + 3)
    
    def extract_tensors(self, experiences:List[experience], experience_index, weight) -> per_experience:
        # Convert batch of Experiences to Experience of batches
        batch = experience(*zip(*experiences))

        t1 = torch.cat(batch.state)
        t2 = torch.cat(batch.next_state)
        t3 = torch.cat(batch.action).squeeze(0).unsqueeze(1)
        t4 = torch.cat(batch.reward).squeeze(0).unsqueeze(1)
        t5 = torch.cat(batch.done).squeeze(0).unsqueeze(1)

        return per_experience(t1,t2,t3,t4,t5, experience_index, weight)