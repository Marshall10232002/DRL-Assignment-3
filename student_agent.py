import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
import gym
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class DuelingDQN(nn.Module):
    def __init__(self, in_channels, n_actions):
        super(DuelingDQN, self).__init__()
        # conv layers: input shape (batch, in_channels=4, 84, 84)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,        64, kernel_size=3, stride=1)
        # flatten & dueling head
        self.fc   = nn.Linear(20736, 512)      # 64 * 18 * 18 = 20736
        self.adv  = nn.Linear(512,  n_actions)
        self.val  = nn.Linear(512,  1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        adv = self.adv(x)
        val = self.val(x)
        # combine streams
        return val + (adv - adv.mean(dim=1, keepdim=True))


class Agent(object):
    """Loads mario_q.pth on CPU; preprocesses raw 240×256×3→84×84×4; acts greedily."""
    def __init__(self):
        # discrete action space: must match COMPLEX_MOVEMENT
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))

        # force CPU
        self.device = torch.device('cpu')

        # build network and load weights
        self.net = DuelingDQN(in_channels=4,
                              n_actions=self.action_space.n).to(self.device)
        ckpt = torch.load("mario_q.pth", map_location=self.device)
        self.net.load_state_dict(ckpt)
        self.net.eval()

        # for stacking last 4 frames
        self.frames = deque(maxlen=4)

    def act(self, observation):
        """
        observation: raw RGB frame shape (240,256,3)
        returns: int action in [0, action_space)
        """
        # 1) to grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        # 2) resize to 84×84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # 3) normalize to [0,1]
        normed = resized.astype(np.float32) / 255.0

        # fill buffer on new episode
        if len(self.frames) < 4:
            for _ in range(4):
                self.frames.append(normed)
        else:
            self.frames.append(normed)

        # stack into shape (4,84,84), then batch → (1,4,84,84)
        state = np.stack(self.frames, axis=0)[None, ...]
        state_t = torch.from_numpy(state).to(self.device)

        # get Q-values and pick best
        with torch.no_grad():
            q = self.net(state_t)
            action = int(q.argmax(dim=1).item())

        return action
