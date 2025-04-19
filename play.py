import os
import cv2
import torch
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

# imports from your training script
from training import DuelingDQN, preprocess_state
from training import RandomStartEnv, RepeatAndMaxEnv, ResizeObservation, NormalizeObservation, ObservationStack

# A wrapper factory that excludes life-based early episode termination
# so that the game runs until all lives are exhausted.
def make_env_full_game(env):
    env = RandomStartEnv(env, max_noop=30)
    env = RepeatAndMaxEnv(env, repeat=4)
    env = ResizeObservation(env, width=84, height=84)
    env = NormalizeObservation(env)
    return ObservationStack(env, k=4)


def main():
    # 1) Build and wrap the environment for full-play
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = make_env_full_game(env)

    # 2) Load your trained DQN
    model = DuelingDQN(4, env.action_space.n, device).to(device)
    checkpoint = torch.load("mario_q.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # 3) Prepare video writer
    obs = env.reset()
    first_frame = env.render(mode="rgb_array")
    height, width, _ = first_frame.shape
    out = cv2.VideoWriter(
        "mario_full_game.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        (width, height)
    )

    # 4) Run until true game over (all lives used)
    total_reward = 0.0
    done = False

    while not done:
        state = preprocess_state(obs)
        with torch.no_grad():
            action = int(model(state).argmax(dim=1).item())

        obs, reward, done, info = env.step(action)
        total_reward += reward

        frame = env.render(mode="rgb_array")
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    print(f"Game over! Total reward across all lives: {total_reward:.2f}")

    # 5) Clean up
    out.release()
    env.close()

if __name__ == "__main__":
    print("Working dir:", os.getcwd())
    print("Available files:", os.listdir())
    main()
