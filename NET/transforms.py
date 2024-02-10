import torchvision.transforms as transforms
from torchvision import transforms as T
import torch
import numpy as np
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
from gym import Wrapper, ObservationWrapper
from collections import deque
    


class GrayScaleObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        # crop observation
        observation = observation[:, 34:34 + 160, :160]
        
        transforms = T.Compose(
            [T.Resize(self.shape, antialias=True), T.Normalize(0, 255)]
        )
        observation = transforms(observation).squeeze(0).numpy()
        return observation


def apply_wrappers(env):
    # Apply Wrappers to environment
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    return env



