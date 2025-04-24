# student_agent.py
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from gym_super_mario_bros import make as make_smb
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# Corrected checkpoint filename to match training script output
CHECKPOINT = "mario_q_noisy_new.pth"
DEVICE     = torch.device("cpu") # Evaluation is typically done on CPU

# ───── NoisyLinear ─────
class NoisyLinear(nn.Module):
    def __init__(self, in_f, out_f, sigma_init=0.5):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.mu_w    = nn.Parameter(torch.empty(out_f, in_f))
        # *** CORRECTED PARAMETER NAME: sigma_w -> sig_w ***
        self.sig_w = nn.Parameter(torch.empty(out_f, in_f))
        self.mu_b    = nn.Parameter(torch.empty(out_f))
        # *** CORRECTED PARAMETER NAME: sigma_b -> sig_b ***
        self.sig_b = nn.Parameter(torch.empty(out_f))
        self.register_buffer("eps_in",  torch.zeros(in_f))
        self.register_buffer("eps_out", torch.zeros(out_f))
        self.reset_parameters(sigma_init)
        self.reset_noise()

    def reset_parameters(self, sigma_init):
        bound = 1 / np.sqrt(self.in_f)
        self.mu_w.data.uniform_(-bound, bound)
        self.mu_b.data.uniform_(-bound, bound)
        # *** CORRECTED PARAMETER NAME: sigma_w -> sig_w ***
        self.sig_w.data.fill_(sigma_init / np.sqrt(self.in_f))
        # *** CORRECTED PARAMETER NAME: sigma_b -> sig_b ***
        self.sig_b.data.fill_(sigma_init / np.sqrt(self.mu_w.size(0))) # Use mu_w size for out_f

    @staticmethod
    def _noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.eps_in.copy_( self._noise(self.in_f) )
        # *** CORRECTED PARAMETER NAME: sigma_w -> sig_w ***
        self.eps_out.copy_( self._noise(self.mu_w.size(0)) ) # Use mu_w size for out_f

    def forward(self, x):
        # Noise is applied here during training (when reset_noise is called)
        # During evaluation (net.eval()), reset_noise is not called, so sig*eps terms are effectively zero,
        # and it behaves like a standard linear layer using only mu_w and mu_b.
        # *** CORRECTED PARAMETER NAME: sigma_w -> sig_w ***
        w = self.mu_w + self.sig_w * self.eps_out.unsqueeze(1) * self.eps_in.unsqueeze(0)
        # *** CORRECTED PARAMETER NAME: sigma_b -> sig_b ***
        b = self.mu_b + self.sig_b * self.eps_out
        return F.linear(x, w, b)

# ───── Dueling DQN ─────
class DuelingDQN(nn.Module):
    def __init__(self, in_frames=4, n_actions=len(COMPLEX_MOVEMENT)):
        super().__init__()
        # *** CORRECTED LAYER NAME: features -> conv ***
        self.conv = nn.Sequential(
            nn.Conv2d(in_frames, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1),        nn.ReLU()
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_frames, 84, 84)
            # *** CORRECTED LAYER NAME: features -> conv ***
            conv_out = self.conv(dummy).view(1, -1).size(1)
        # Using NoisyLinear for the final layers
        self.fc  = NoisyLinear(conv_out, 512)
        self.adv = NoisyLinear(512, n_actions)
        self.val = NoisyLinear(512, 1)

    def reset_noise(self):
        # This is called during training for exploration
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        # *** CORRECTED LAYER NAME: features -> conv ***
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        adv = self.adv(x)
        val = self.val(x)
        return val + adv - adv.mean(1, keepdim=True)


class Agent:
    def __init__(self):
        self.net = DuelingDQN().to(DEVICE)
        # Load the state dict, ensuring map_location is correct
        # The FutureWarning is just a warning about pickle security, not the cause of the error.
        # You can ignore it for now or set weights_only=True if you trust the source of the file.
        self.net.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        self.net.eval() # Set the network to evaluation mode (disables dropout, batchnorm stats, and importantly, NoisyLinear noise)

        # Repeat & Max logic (repeat=4)
        self.skip = 4 # <--- Agent decides every 4 raw frames
        self.skip_ctr = 0 # <--- Counter for raw frames since last decision
        # pool_buf stores raw frames (240x256x3) for max-pooling
        self.pool_buf = deque(maxlen=2)

        # Frame stack logic (k=4)
        # stack_buf stores processed 84x84 uint8 frames
        self.stack_buf = deque(maxlen=4)
        self.last_act = 0 # Store the last action taken by the agent network
        # Random initial skip logic (handled manually in evaluation agent)
        self.rand = 0
        self.randskip = 0

    def act(self, obs):
        # Add current raw frame to max-pool buffer
        self.pool_buf.append(obs)

        # Handle random initial skip (returns last_act without processing/network)
        if self.randskip < self.rand:
            self.randskip += 1
            # Process and stack the frame even during no-ops
            frame = np.maximum(self.pool_buf[-2], self.pool_buf[-1]) if len(self.pool_buf) == 2 else obs
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            proc = small.astype(np.uint8)
            self.stack_buf.append(proc)
            while len(self.stack_buf) < 4: self.stack_buf.append(proc) # Pad
            return self.last_act # Returns the repeated action

        # Handle frame skipping
        self.skip_ctr += 1 # <--- Increments counter for *this* raw frame
        # Not yet time to take a new action → repeat last action
        if self.skip_ctr < self.skip: # <--- If not 4 raw frames yet
            # Process and stack the frame even during skip steps
            frame = np.maximum(self.pool_buf[-2], self.pool_buf[-1]) if len(self.pool_buf) == 2 else obs
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            proc = small.astype(np.uint8)
            self.stack_buf.append(proc)
            while len(self.stack_buf) < 4: self.stack_buf.append(proc) # Pad
            return self.last_act # Returns the repeated action

        # Time to act: reset skip counter
        self.skip_ctr = 0 # <--- Resets counter every 4th raw frame

        # Max-pool last two frames (pool_buf already has the current obs)
        frame = np.maximum(self.pool_buf[-2], self.pool_buf[-1]) if len(self.pool_buf) == 2 else obs  # fallback (very first step)

        # Grayscale + resize
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        proc = small.astype(np.uint8)

        # Add to frame stack
        self.stack_buf.append(proc)
        while len(self.stack_buf) < 4: # Pad if needed (first few steps)
            self.stack_buf.append(proc)

        # Construct input tensor (shape: 1x4x84x84)
        state = np.stack(list(self.stack_buf), axis=0)[None] # Add batch dimension
        state = torch.tensor(state, dtype=torch.float32, device=DEVICE) / 255.0

        # Forward pass (net.eval() ensures no noise and uses mean weights)
        with torch.no_grad():
            q = self.net(state)

        self.last_act = int(q.argmax(1).item()) # <--- Selects a *new* action
        return self.last_act # Returns the newly selected action


if __name__ == "__main__":
    env = JoypadSpace(make_smb("SuperMarioBros-v0"), COMPLEX_MOVEMENT)
    agent = Agent()
    obs, done = env.reset(), False
    total = 0.0

    # Get frame size for VideoWriter
    height, width, _ = obs.shape
    video_path = "mario_eval.mp4"
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),  # codec
        60.0,                             # FPS
        (width, height)                  # frame size
    )
    stuck, prev_x=0,0
    # --- TA's Specified Evaluation Loop ---
    while not done:
        writer.write(cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))  # write frame in BGR
        a = agent.act(obs) # <--- Calls agent.act() with current obs, gets ONE action
        obs, r, done, info = env.step(a) # <--- Steps environment *once* with that action
        cur_x =info.get('x_pos', prev_x)
        stuck = stuck + 1 if cur_x - prev_x <= 0 else 0
        prev_x = cur_x
        if stuck >= 600: done = True
        total += r # <--- Accumulates reward from that single step

    writer.release()
    env.close()
    print(f"Episode reward: {total:.1f}")
    print(f"Saved video to: {video_path}")
    # --- End TA's Specified Evaluation Loop ---