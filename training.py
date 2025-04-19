import warnings
warnings.filterwarnings(
    "ignore",
    message=".*out of date.*",
    category=UserWarning,
    module="gym.envs.registration"
)


import os
import pickle
import random
import time
from collections import deque

import cv2
import gym
import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

# ------------------ Configuration ------------------

# Set to True to load saved weights and continue training;
# set to False to train from scratch.
resume_training = False

# ------------------ Environment Wrappers ------------------

class RandomStartEnv(gym.Wrapper):
    """Perform a random number of NOOPs at the start of each episode."""
    def __init__(self, env, max_noop=30):
        super().__init__(env)
        self.max_noop = max_noop
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(random.randint(1, self.max_noop)):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs
    def step(self, action):
        return self.env.step(action)

class LifeResetEnv(gym.Wrapper):
    """End episode on life loss but reset game only when all lives are gone."""
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self._real_done = True
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._real_done = done
        lives = info.get('life', getattr(self.env.unwrapped, '_life', 0))
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info
    def reset(self, **kwargs):
        if self._real_done:
            obs = self.env.reset(**kwargs)
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = getattr(self.env.unwrapped, '_life', 0)
        return obs

class RepeatAndMaxEnv(gym.Wrapper):
    """Repeat action for N frames and max-pool last two observations."""
    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat
        self.buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
    def step(self, action):
        total_reward, done = 0.0, False
        for i in range(self.repeat):
            obs, r, done, info = self.env.step(action)
            if i >= self.repeat - 2:
                self.buffer[i - (self.repeat - 2)] = obs
            total_reward += r
            if done:
                break
        max_frame = self.buffer.max(axis=0)
        return max_frame, total_reward, done, info
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ResizeObservation(gym.ObservationWrapper):
    """Resize frames to 84×84 grayscale."""
    def __init__(self, env, width=84, height=84):
        super().__init__(env)
        self.width, self.height = width, height
        self.observation_space = spaces.Box(0, 255, (height, width, 1), dtype=np.uint8)
    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, -1)

class NormalizeObservation(gym.ObservationWrapper):
    """Scale pixel values to [0,1]."""
    def __init__(self, env):
        super().__init__(env)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(0.0, 1.0, shp, dtype=np.float32)
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0

class ObservationStack(gym.Wrapper):
    """Stack the last k observations along the channel dimension."""
    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            0, 255, (shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype
        )
    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._stack()
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._stack(), reward, done, info
    def _stack(self):
        return np.concatenate(list(self.frames), axis=-1)

def make_env(env):
    env = RandomStartEnv(env)
    env = RepeatAndMaxEnv(env)
    env = LifeResetEnv(env)
    env = ResizeObservation(env)
    env = NormalizeObservation(env)
    return ObservationStack(env, k=4)


# ------------------ Replay Buffer ------------------

class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, transition):
        self.buffer.append(transition)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)


# ------------------ Dueling DQN Network ------------------

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)

class DuelingDQN(nn.Module):
    def __init__(self, n_frames, n_actions, device):
        super().__init__()
        self.conv1 = nn.Conv2d(n_frames, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc   = nn.Linear(20736, 512)
        self.adv  = nn.Linear(512, n_actions)
        self.val  = nn.Linear(512, 1)
        self.apply(init_weights)
        self.device = device

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x).to(self.device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 20736)
        x = F.relu(self.fc(x))
        advantage = self.adv(x)
        value     = self.val(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


# ------------------ Helpers & Training ------------------

def preprocess_state(frame):
    """From H×W×C to 1×C×H×W"""
    arr = np.array(frame).transpose(2, 0, 1)
    return np.expand_dims(arr, 0)

def train_step(policy_net, target_net, buffer, batch_sz, discount_factor, optimizer, device):
    batch = buffer.sample(batch_sz)
    states, rewards, actions, next_states, masks = zip(*batch)

    states      = np.array(states).squeeze()
    next_states = np.array(next_states).squeeze()
    rewards     = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
    masks       = torch.FloatTensor(masks).unsqueeze(-1).to(device)

    with torch.no_grad():
        next_actions = policy_net(next_states).max(1)[1].unsqueeze(-1)
        q_next       = target_net(next_states).gather(1, next_actions)
        target_q     = rewards + discount_factor * q_next * masks

    actions_t = torch.tensor(actions).unsqueeze(-1).to(device)
    q_current = policy_net(states).gather(1, actions_t.long())

    loss = F.smooth_l1_loss(q_current, target_q).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def sync_networks(policy_net, target_net):
    target_net.load_state_dict(policy_net.state_dict())


# ------------------ Main Training Loop ------------------

def main():
    # build and wrap environment
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = make_env(env)

    # device & networks
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    policy_net  = DuelingDQN(4, env.action_space.n, device).to(device)
    target_net  = DuelingDQN(4, env.action_space.n, device).to(device)
    sync_networks(policy_net, target_net)
    optimizer   = optim.Adam(policy_net.parameters(), lr=1e-4)
    print("Using device:", device)

    # optionally resume from checkpoint
    if resume_training:
        if os.path.exists("mario_q.pth") and os.path.exists("mario_q_target.pth"):
            policy_net.load_state_dict(torch.load("mario_q.pth", weights_only=True))
            target_net.load_state_dict(torch.load("mario_q_target.pth", weights_only=True))
            print("✔ Loaded checkpoint, continuing training.")
        else:
            print("⚠ No checkpoint found, starting fresh.")

    # hyperparameters
    discount_factor     = 0.99
    batch_sz            = 256
    buffer_capacity     = 50000
    epsilon             = 0.001
    target_update_freq  = 50
    log_interval        = 10

    # buffers and trackers
    replay_buffer   = ExperienceBuffer(buffer_capacity)
    train_counter   = 0
    reward_accum    = 0.0
    loss_accum      = 0.0
    score_history   = []
    start_time      = time.perf_counter()

    for episode in range(10000):
        state       = preprocess_state(env.reset())
        done        = False
        stuck_steps = 0
        prev_x      = 0
        step_thresh = 150

        while not done:
            # ε-greedy action
            if epsilon > random.random():
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(policy_net(state).argmax(dim=1).cpu().item())

            raw_next, reward_raw, done, info = env.step(action)

            # detect stuck and penalize
            cur_x    = info.get('x_pos', prev_x)
            delta_x  = cur_x - prev_x
            prev_x   = cur_x
            if delta_x <= 0:
                stuck_steps += 1
            else:
                stuck_steps = 0
            if stuck_steps >= step_thresh:
                reward_raw += -0.1

            next_state = preprocess_state(raw_next)
            shaped_r   = np.sign(reward_raw) * (np.sqrt(abs(reward_raw) + 1) - 1) + 0.001 * reward_raw

            replay_buffer.push((state, float(shaped_r), action, next_state, 1 - int(done)))
            state = next_state
            reward_accum += reward_raw

            if len(replay_buffer) > 2000:
                loss = train_step(policy_net, target_net,
                                  replay_buffer, batch_sz,
                                  discount_factor, optimizer, device)
                loss_accum += loss
                train_counter += 1
                if train_counter % target_update_freq == 0:
                    sync_networks(policy_net, target_net)
                    torch.save(policy_net.state_dict(), "mario_q.pth")
                    torch.save(target_net.state_dict(), "mario_q_target.pth")

        # logging
        if episode % log_interval == 0:
            elapsed     = time.perf_counter() - start_time
            avg_score   = reward_accum / log_interval
            avg_loss    = loss_accum   / log_interval
            stage_info  = getattr(env.unwrapped, "_stage", None)
            print(f"{device} | Ep {episode} | Avg Score {avg_score:.2f} | "
                  f"Avg Loss {avg_loss:.2f} | Stage {stage_info} | Time {elapsed:.1f}s")
            score_history.append(avg_score)
            pickle.dump(score_history, open("score.p", "wb"))

            reward_accum = 0.0
            loss_accum   = 0.0
            start_time   = time.perf_counter()

if __name__ == "__main__":
    main()
