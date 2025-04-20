import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class DuelingDQN(nn.Module):
    def __init__(self, in_channels=4, n_actions=len(COMPLEX_MOVEMENT)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc   = nn.Linear(64 * 18 * 18, 512)
        self.adv  = nn.Linear(512, n_actions)
        self.val  = nn.Linear(512, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        # x: tensor (batch, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        adv = self.adv(x)
        val = self.val(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

class Agent:
    def __init__(self, model_path="mario_q_randseq.pth", skip=4):
        # device and network setup
        self.device = torch.device('cpu')
        self.net = DuelingDQN().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device)
        self.net.load_state_dict(ckpt)
        self.net.eval()

        # frame skip and pooling
        self.skip = skip
        self.skip_buffer = deque(maxlen=skip)

        # history of processed frames
        self.processed_buffer = deque(maxlen=4)

        # last chosen action (default NOOP = 0)
        self.last_action = 0

    def act(self, obs, epsilon=0.001):
        """
        obs: raw RGB frame (240×256×3) from env.reset() or env.step()
        returns: single int action
        """
        # collect raw frame
        self.skip_buffer.append(obs)

        # if not enough frames to pool, repeat last action
        if len(self.skip_buffer) < self.skip:
            return self.last_action

        # max‑pool last two frames
        f1, f2 = self.skip_buffer[-2], self.skip_buffer[-1]
        max_frame = np.maximum(f1, f2)

        # preprocess: grayscale → resize → normalize
        gray = cv2.cvtColor(max_frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        proc = resized.astype(np.float32) / 255.0

        # update processed frame history
        self.processed_buffer.append(proc)
        # pad initial history if needed
        if len(self.processed_buffer) < 4:
            while len(self.processed_buffer) < 4:
                self.processed_buffer.append(proc)

        # create state tensor (1,4,84,84)
        state = np.stack(self.processed_buffer, axis=0)[None, ...]
        state_t = torch.from_numpy(state).to(self.device)

        # forward pass
        with torch.no_grad():
            q = self.net(state_t)
            action = int(q.argmax(dim=1).item())

        # save and reset skip buffer
        self.last_action = action
        self.skip_buffer.clear()

        return action
