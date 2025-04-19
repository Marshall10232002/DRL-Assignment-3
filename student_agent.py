import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import deque
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


class DuelingDQN(nn.Module):
    def __init__(self, n_frames: int, n_actions: int):
        super().__init__()
        # conv layers (input [B, n_frames, 84, 84])
        self.conv1 = nn.Conv2d(n_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32,    64, kernel_size=3, stride=1)
        # flatten & dueling head; conv2 outputs 64×18×18 = 20736
        self.fc   = nn.Linear(20736, 512)
        self.adv  = nn.Linear(512, n_actions)
        self.val  = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        adv = self.adv(x)
        val = self.val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class Agent(object):
    """
    - loads your trained `mario_q.pth` onto CPU
    - on act(): converts raw RGB (240×256×3)
      → gray → 84×84 → [0,1] float → stacks last 4 → feeds into DuelingDQN
    """
    def __init__(self):
        # match the COMPLEX_MOVEMENT space
        self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))

        # CPU only
        self.device = torch.device('cpu')

        # build network & load weights
        self.policy_net = DuelingDQN(n_frames=4,
                                    n_actions=self.action_space.n).to(self.device)
        ckpt = torch.load('mario_q.pth', map_location=self.device)
        self.policy_net.load_state_dict(ckpt)
        self.policy_net.eval()

        # for stacking frames
        self.frames = deque(maxlen=4)

    def _preprocess(self, obs: np.ndarray) -> np.ndarray:
        gray    = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)[None, ...]

    def act(self, observation):
        # get one preprocessed frame → (1,84,84)
        frame = self._preprocess(observation)

        # fill buffer on new episode
        if len(self.frames) < 4:
            self.frames.extend([frame] * (4 - len(self.frames)))
        self.frames.append(frame)

        # stack & batch → (1,4,84,84)
        state = np.concatenate(self.frames, axis=0)[None, ...]
        state_t = torch.from_numpy(state).to(self.device)

        with torch.no_grad():
            q = self.policy_net(state_t)
            a = int(q.argmax(dim=1).item())

        return a
