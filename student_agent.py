import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# ——————— Factorised Gaussian NoisyLinear ———————
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

# ——————— Noisy‑Net Dueling DQN ———————
class DuelingDQN(nn.Module):
    def __init__(self, input_frames=4, n_actions=len(COMPLEX_MOVEMENT)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_frames, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),           nn.ReLU()
        )
        # infer conv‐output size
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
        x = x / 1.0            # assume already in [0,1]
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

# ——————— Agent for evaluation ———————
class Agent:
    def __init__(self):
        self.device = torch.device('cpu')
        self.net = DuelingDQN(4, len(COMPLEX_MOVEMENT)).to(self.device)
        self.net.load_state_dict(
            torch.load("mario_q_noisy.pth", map_location=self.device)
        )
        self.net.eval()

        self.skip_buffer  = deque(maxlen=4)
        self.frame_stack  = deque(maxlen=4)
        self.last_action  = 0

    def act(self, obs):
        # 1) frame‐skip pooling (repeat=4)
        self.skip_buffer.append(obs)
        if len(self.skip_buffer) < 4:
            return self.last_action
        f1, f2 = self.skip_buffer[-2], self.skip_buffer[-1]
        max_frame = np.maximum(f1, f2)

        # 2) preprocess: gray → resize → normalize
        gray    = cv2.cvtColor(max_frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        proc    = resized.astype(np.float32) / 255.0

        # 3) stack into 4‐frame state
        self.frame_stack.append(proc)
        while len(self.frame_stack) < 4:
            self.frame_stack.append(proc)
        state = np.stack(self.frame_stack, axis=0)[None, ...]
        state_t = torch.from_numpy(state).to(self.device)

        # 4) noisy‐net forward
        self.net.reset_noise()
        with torch.no_grad():
            q = self.net(state_t)
            action = int(q.argmax(dim=1).item())

        self.last_action = action
        self.skip_buffer.clear()
        return action
