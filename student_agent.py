import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from gym import spaces

# ================== Environment wrappers ==================
class RandomStartEnv(gym.Wrapper):
    """Insert random NOOPs at episode start"""
    def __init__(self, env, max_noop=30):
        super().__init__(env)
        self.max_noop = max_noop
        self.noop_action = 0
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(random.randint(1, self.max_noop)):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

class LifeResetEnv(gym.Wrapper):
    """Treat life loss as episode end"""
    def __init__(self, env):
        super().__init__(env)
        self.lives, self._real_done = 0, True
    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self._real_done = done
        lives = info.get('life', getattr(self.env.unwrapped, '_life', 0))
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, r, done, info
    def reset(self, **kwargs):
        if self._real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, *_ = self.env.step(0)
        self.lives = getattr(self.env.unwrapped, '_life', 0)
        return obs

class RepeatAndMaxEnv(gym.Wrapper):
    """Repeat action & max‐pool last two frames"""
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat
        shp = env.observation_space.shape
        self.buffer = np.zeros((2,)+shp, dtype=np.uint8)
    def step(self, action):
        total_r, done = 0.0, False
        for i in range(self.repeat):
            obs, r, done, info = self.env.step(action)
            if i >= self.repeat - 2:
                self.buffer[i - (self.repeat - 2)] = obs
            total_r += r
            if done:
                break
        max_frame = self.buffer.max(axis=0)
        return max_frame, total_r, done, info

class ResizeObservation(gym.ObservationWrapper):
    """84×84 grayscale"""
    def __init__(self, env, w=84, h=84):
        super().__init__(env)
        self.w, self.h = w, h
        self.observation_space = spaces.Box(0, 255, (h, w, 1), np.uint8)
    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, -1)

class ObservationStack(gym.Wrapper):
    """Stack k frames on channel axis"""
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(0, 255, (h, w, c*k), np.uint8)
        self.frames = deque(maxlen=k)
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=-1)
    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=-1), r, done, info

# ========== Monkey‐patch gym_super_mario_bros.make ==========
_original_make = gym_super_mario_bros.make

def _wrapped_make(*args, **kwargs):
    env = _original_make(*args, **kwargs)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = RandomStartEnv(env)
    env = RepeatAndMaxEnv(env)
    env = LifeResetEnv(env)
    env = ResizeObservation(env)
    env = ObservationStack(env, k=4)
    return env

gym_super_mario_bros.make = _wrapped_make

# =========== NoisyNet & DuelingDQN definitions ===========
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

class DuelingDQN(nn.Module):
    def __init__(self, input_frames=4, n_actions=len(COMPLEX_MOVEMENT)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_frames, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),           nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_frames, 84, 84)
            conv_out = self.features(dummy).view(1, -1).size(1)
        self.fc  = NoisyLinear(conv_out, 512)
        self.adv = NoisyLinear(512, n_actions)
        self.val = NoisyLinear(512, 1)
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.reset_noise()
    def forward(self, x):
        x = x.float() / 1.0
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - adv.mean(dim=1, keepdim=True)

# =================== Agent class ===================
class Agent:
    def __init__(self, model_path="mario_q_noisy.pth", device=None):
        self.device = device or torch.device('cpu')
        self.net = DuelingDQN().to(self.device)
        self.net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.net.eval()
        self.skip        = 4
        self._skip_count = 0
        self.pool_buf    = deque(maxlen=2)
        self.stack_buf   = deque(maxlen=4)
        self.last_action = 0

    def reset(self):
        self._skip_count = 0
        self.pool_buf.clear()
        self.stack_buf.clear()
        self.last_action = 0

    def act(self, obs):
        self.pool_buf.append(obs)
        action = self.last_action
        if self._skip_count == self.skip - 1:
            if len(self.pool_buf) == 2:
                frame = np.maximum(self.pool_buf[0], self.pool_buf[1])
            else:
                frame = obs
            gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA)
            proc    = resized.astype(np.float32) / 255.0
            self.stack_buf.append(proc)
            while len(self.stack_buf) < 4:
                self.stack_buf.append(proc)
            state = np.stack(self.stack_buf, axis=0)[None,...]
            state_t = torch.from_numpy(state).to(self.device)
            with torch.no_grad():
                q = self.net(state_t)
            action = int(q.argmax(dim=1).item())
            self.last_action = action
        self._skip_count = (self._skip_count + 1) % self.skip
        return action
