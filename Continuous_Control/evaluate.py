"""
Step through a trained continuous-action controller with the text-ruler
visualization from environment.py's render(), so you can watch the mass
get pulled back to x=0 in real time (well, in printed-frame time).
"""
import argparse

from environment import PointMassEnv
from agent import ContinuousACAgent

CHECKPOINT_PATH = "./Continuous_Control/checkpoints/continuous_ac_pointmass.pt"


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--x0", type=float, default=3.0)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    return args


def evaluate(checkpoint_path=CHECKPOINT_PATH, x0=3.0, deterministic=True):
    env = PointMassEnv(seed=0)
    agent = ContinuousACAgent(state_size=env.n_states, action_size=1)
    agent.load(checkpoint_path)
    if deterministic:
        agent.std = 1e-6

    state = env.reset(x0=x0)
    env.render()
    total_reward = 0.0
    done = False

    while not done:
        u, _ = agent.select_action(state)
        state, reward, done = env.step(u)
        total_reward += reward
        env.render()

    print(f"\nTotal reward: {total_reward:.2f}  Final x: {state[0]:+.3f}  Final v: {state[1]:+.3f}")


if __name__ == "__main__":
    args = init()
    evaluate(checkpoint_path=args.checkpoint, x0=args.x0, deterministic=not args.stochastic)