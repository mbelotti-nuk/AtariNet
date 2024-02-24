import torchvision.transforms as transforms
from torchvision import transforms as T
import numpy as np
import gym
from collections import deque
from matplotlib import pyplot as plt
    
# Class to convert images to grayscale and crop
class Transforms:
    def to_gray(frame):
        gray_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop((175,150)),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        new_frame = gray_transform(frame)

        return new_frame.numpy()

class Environment:
    def __init__(self, env, skip_frames, n_frames, state_space):
        self.env = env

        self._state = deque(np.zeros(state_space), maxlen=n_frames)
        self._next_state = deque(np.zeros(state_space), maxlen=n_frames)
        self._stack = deque(maxlen=2)
        self._n_lifes = 5
        
        self.black_screen = np.zeros((state_space[1],state_space[2]))

        self._skip_frames = skip_frames
        self._n_frames = n_frames
    
    def reset(self):
        
        self._n_lifes = 5

        # Reset environment and preprocess state
        obs, _  = self.env.reset()
        state = Transforms.to_gray(obs[0])
        # initialize state
        self._state = deque( [state[0]]*self._n_frames, maxlen=self._n_frames )
        # initialize next_state
        self._next_state = self._state.copy()
        # initialize stack
        self._stack.append(obs[0])

    def step(self, action):
        # initialize reward
        reward = 0 
        # do k consecutive step with the same action
        for k in range(self._skip_frames):
            obs_, action_reward, done, trunc, info = self.env.step(action)
            reward += action_reward 
            self._stack.append(obs_)
            if done: break

        self._next_state = self._state.copy()

        # if dead, reset the history, since previous states don't matter anymore
        if done: 
             self._next_state.append(self.black_screen)
        else:
            # blur last two frames
            next_obs = np.maximum(*list(self._stack))
            # transform the observation
            state_ = Transforms.to_gray(next_obs).squeeze(0)
            # append the last observation, skipping n=SKIP_ACTIONS that were before
            self._next_state.append(state_)


        # update reward when losing game
        if info['lives'] < self._n_lifes:
            reward += -2
            # re-assign n_lifes
            self._n_lifes = info['lives'] 

        # clip reward
        reward = np.sign(reward)

        return np.array(self._state), np.array(self._next_state), action, reward, int(done)

    def state(self):
        # re-assign state
        self._state = self._next_state.copy()      
        return self._state
    
    def debug_imgs(self): 
        for i in range(0, len(self._next_state)):        
            plt.imshow(self._next_state[i], cmap="gray")
            plt.savefig(f"next_state_{i}.png")
            plt.clf()
            plt.imshow(self._state[i], cmap="gray")
            plt.savefig(f"state_{i}.png")
            plt.clf()
        return



