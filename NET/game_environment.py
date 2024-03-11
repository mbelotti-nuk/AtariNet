import torchvision.transforms as transforms
from torchvision import transforms as T
import numpy as np
import gym
from collections import deque
from matplotlib import pyplot as plt
from PIL import Image
    
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
    def __init__(self, env, skip_frames, n_frames, state_space, train = True, noop_max=5, save_video =False):
        self.env = env

        self._state = deque(np.zeros(state_space), maxlen=n_frames)
        self._next_state = deque(np.zeros(state_space), maxlen=n_frames)
        self._stack = deque(maxlen=2)
        self._n_lifes = 5

        self.train = train
        self.noop_max = noop_max
        self.noop_action = 0
        self.fire_action = 1
        
        self.black_screen = np.zeros((state_space[1],state_space[2]))

        self._skip_frames = skip_frames
        self._n_frames = n_frames

        self.save_video = save_video
        self.frames = []

        self._end_game = False
    
    @property
    def end_game(self):
        return self._end_game

    def reset(self):
        self._n_lifes = 5
        self._end_game = False

        # initialize state
        self._state = deque([self.black_screen]*self._n_frames, maxlen=self._n_frames)
        
        self._get_reset()

        # initialize next state
        self._next_state = self._state.copy()
        #self._stack.append(obs[0])

    def _get_reset(self):
        # Reset environment and preprocess state
        obs, _  = self.env.reset()
        obs = obs[0]

        # fire re-set if testing
        if not self.train:
            obs, _, done, _, _ = self.env.step(self.fire_action)
            if done:
                self.env.reset()
            obs, _, done, _, _ = self.env.step(2)
            if done:
                obs = self.env.reset()
                obs = obs[0]
            if self.save_video:
                self.frames.append(obs)
            state = Transforms.to_gray(obs).squeeze(0)
            self._state.append(state)
            return 
        else:
            noops = np.random.randint(1, self.noop_max + 1)  
            assert noops > 0
            for _ in range(noops):
                obs, _, done, _, _ = self.env.step(self.noop_action)
                if done:
                    obs = self.env.reset()
                    obs = obs[0]
            state = Transforms.to_gray(obs).squeeze(0)
            self._state.append(state)       
            return


    def step(self, action):
        # initialize reward
        reward = 0 
        # do k consecutive step with the same action
        for k in range(self._skip_frames):
            obs_, action_reward, done, _, info = self.env.step(action)
            reward += action_reward 
            self._stack.append(obs_)
            if done: 
                self._end_game = True
                break
            if self.save_video:
                self.frames.append(obs_)
        
        # get next state
        self.get_next_state()

        # if end-game or life lost, add black screen
        if done: 
             self._next_state.append(self.black_screen)
        elif info['lives'] < self._n_lifes:
            self._next_state.append(self.black_screen)
            self._n_lifes = info['lives']
            # add flag
            done = True
            #reward = -1

        # clip reward
        reward = np.sign(reward)

        return np.array(self._state), np.array(self._next_state), action, reward, int(done)


    def get_next_state(self):
        self._next_state = self._state.copy()
        # blur last two frames
        next_obs = np.maximum(*list(self._stack))
        # transform the observation
        state_ = Transforms.to_gray(next_obs).squeeze(0)
        # append the last observation, skipping n=SKIP_ACTIONS that were before
        self._next_state.append(state_)

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
    
    # Save gif of an episode starting num_transitions ago from memory
    def save_gif(self):
        frames = [Image.fromarray(image, mode='RGB')for image in self.frames]
        frame_one = frames[0]
        frame_one.save("match.gif", format="GIF", append_images=frames,
                save_all=True, duration=100, loop=0)



