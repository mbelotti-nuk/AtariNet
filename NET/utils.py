import numpy as np
from collections import namedtuple
from dataclasses import dataclass
import torch


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    pointer = 0

    def __init__(self, capacity):
        # number of leaf nodes
        self.capacity = capacity
        # initialize tree
        self.tree = np.zeros(2 * capacity - 1)
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0


    # store priority and sample
    def add(self, priority:int, data):
        idx = self.pointer + self.capacity - 1

        """ tree:
                0
               / \
              0   0
             / \ / \
           idx 0 0  0  We fill the leaves from left to right
        """
        
        self.data[self.pointer] = data
        # update leaf
        self.update(idx, priority)

        self.pointer += 1
        if self.pointer >= self.capacity:
            self.pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, priority):
        # difference between new priority score 
        # and old priority score
        change = priority - self.tree[idx]

        # update priority
        self.tree[idx] = priority
        self._propagate(idx, change)

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # get priority and sample
    def get_leaf(self, s):
        leaf_index = self._retrieve(0, s)
        data_index = leaf_index - self.capacity + 1
        return (leaf_index, self.tree[leaf_index], self._get_data(data_index))

    def _get_data(self, index):
        return [torch.tensor(self.data[index][0]),
                torch.tensor(self.data[index][1]),
                torch.tensor(self.data[index][2]),
                torch.tensor(self.data[index][3]),
                torch.tensor(self.data[index][4])]

    # find sample on leaf node
    def _retrieve(self, parent_index, s):
        left_child_index  = 2 * parent_index + 1
        right_child_index = left_child_index + 1

        if left_child_index >= len(self.tree):
            return parent_index

        if s <= self.tree[left_child_index]:
            return self._retrieve(left_child_index, s)
        else:
            return self._retrieve(right_child_index, s - self.tree[left_child_index])
        
    @property
    def total_priority(self):
        return self.tree[0]

