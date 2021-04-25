from abc import ABC, abstractmethod
import numpy as np
import torch


class ReplayBuffer(ABC):
    @abstractmethod
    def store(self, state, action, reward, next_state ,done):
        pass
        
    @abstractmethod
    def sample(self, batch_size):
        pass


class ExperienceReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        self.buffer = []
        self.capacity = int(size)
        self.pos = 0
    
    def store(self, state, action, reward ,next_state ,done ,skill_one_hot):
        self._add((state, action, reward ,next_state ,done,skill_one_hot))

    def sample(self, batch_size):
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        minibatch = [self.buffer[key] for key in keys]
        return minibatch

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)


