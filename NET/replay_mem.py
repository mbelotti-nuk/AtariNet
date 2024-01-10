import random
from collections import namedtuple, deque
import numpy as np

# Create a named tuple to more semantically transform transitions and batches of transitions
transition = namedtuple('transition', ("state", "action", "reward", "next_state", "done", "raw_state"))

# Memory which allows for storing and sampling batches of transitions
class ReplayBuffer(object):
    def __init__(self, size=1e6):
        self.replay_memory_size = 1_000_000
        self.buffer = deque(maxlen=self.replay_memory_size)
        self.max_size = size

    # Adds a single transitions to the memory buffer
    def add_transition(self, *args):        
        self.buffer.append(transition(*args))

    # Samples a batch of transitions
    def sample_batch(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)

        # Converts batch of transitions to transitions of batches
        batch = transition(  np.array([b.state for b in batch]), np.array([b.action for b in batch], dtype=np.int64), np.array([b.reward for b in batch]), 
                             np.array([b.next_state for b in batch]), np.array([b.done for b in batch]), np.array([b.raw_state for b in batch])  )

        return batch
      

    def __len__(self):
        return len(self.buffer)
    
