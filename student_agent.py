import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from collections import deque

# ——— Factorised Gaussian NoisyLinear —————————————
class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma_init=0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.mu_w    = nn.Parameter(torch.empty(out_f, in_f))
        self.sigma_w = nn.Parameter(torch.empty(out_f, in_f))
        self.mu_b    = nn.Parameter(torch.empty(out_f))
        self.sigma_b = nn.Parameter(torch.empty(out_f))
        self.register_buffer('eps_in',  torch.zeros(in_f))
        self.register_buffer('eps_out', torch.zeros(out_f))
        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        bound = 1 / np.sqrt(self.in_f)
        self.mu_w.data.uniform_(-bound, bound)
        self.mu_b.data.uniform_(-bound, bound)
        self.sigma_w.data.fill_(sigma_init / np.sqrt(self.in_f))
        self.sigma_b.data.fill_(sigma_init / np.sqrt(self.out_f))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.eps_in.copy_( self._scale_noise(self.in_f) )
        self.eps_out.copy_( self._scale_noise(self.out_f) )

    def forward(self, x):
        w_noise = self.eps_out.unsqueeze(1) * self.eps_in.unsqueeze(0)
        w = self.mu_w + self.sigma_w * w_noise
        b = self.mu_b + self.sigma_b * self.eps_out
        return F.linear(x, w, b)

# ——— Noisy‑Net Dueling DQN ——————————————————————
class DuelingDQN(nn.Module):
    def __init__(self, input_frames=4, n_actions=len(COMPLEX_MOVEMENT)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_frames, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),           nn.ReLU()
        )
        # infer conv output dim
        with torch.no_grad():
            dummy = torch.zeros(1, input_frames, 84, 84)
            conv_out_dim = self.features(dummy).view(1, -1).size(1)

        self.fc  = NoisyLinear(conv_out_dim, 512)
        self.adv = NoisyLinear(512, n_actions)
        self.val = NoisyLinear(512, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0.01)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        x = x / 1.0            # assume x in [0,1]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

# ——— Agent for evaluation ——————————————————————
class Agent:
    def __init__(self, model_path="mario_q_noisy.pth", device=None):
        # load model
        self.device = device or torch.device('cpu')
        self.net = DuelingDQN(4, len(COMPLEX_MOVEMENT)).to(self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()

        # skip/pool/stack params
        self.skip        = 4
        self._skip_count = 0
        self.pool_buf    = deque(maxlen=2)
        self.stack_buf   = deque(maxlen=4)
        self.last_action = 0

    def reset(self):
        """Call after env.reset() at start of each episode."""
        self._skip_count = 0
        self.pool_buf.clear()
        self.stack_buf.clear()
        self.last_action = 0

    def act(self, obs):
        """
        obs: raw RGB frame array (H×W×3)
        return: action int
        """
        # 1) add to 2-frame pool
        self.pool_buf.append(obs)

        # 2) default–repeat last
        action = self.last_action

        # 3) on the 4th call (skip_count==3), pool & compute new
        if self._skip_count == self.skip - 1:
            # max-pool last 2 frames
            if len(self.pool_buf) == 2:
                f2, f3 = self.pool_buf[0], self.pool_buf[1]
                frame = np.maximum(f2, f3)
            else:
                frame = obs

            # preprocess: gray→84×84→[0,1]
            gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
            proc    = resized.astype(np.float32) / 255.0

            # build 4-frame stack
            self.stack_buf.append(proc)
            while len(self.stack_buf) < 4:
                self.stack_buf.append(proc)
            state = np.stack(self.stack_buf, axis=0)[None,...]  # [1,4,84,84]
            state_t = torch.from_numpy(state).to(self.device)

            # deterministic forward
            with torch.no_grad():
                q = self.net(state_t)
            action = int(q.argmax(dim=1).item())
            self.last_action = action

        # 4) advance skip count mod skip
        self._skip_count = (self._skip_count + 1) % self.skip
        return action
    
