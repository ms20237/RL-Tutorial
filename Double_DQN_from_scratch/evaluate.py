"""
Same idea as Project 3: load a checkpoint, run it greedily, report
performance. Only the import and default checkpoint path differ.
"""

import argparse

import numpy as np

from environment import CartPoleEnv
from agent import DoubleDQNAgent

CHECKPOINT_PATH = "./Double_DQN_from_scratch/checkpoints/double_dqn_cartpole.pt"


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()
    return args


def evaluate(checkpoint_path=CHECKPOINT_PATH, n_episodes=10, render=False, max_steps=500):
    env = CartPoleEnv(render=render)
    agent = DoubleDQNAgent(n_states=env.n_states, n_actions=env.n_actions)
    agent.load(checkpoint_path)
    agent.epsilon = 0.0

    episode_rewards = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            state = next_state
            total_reward += reward
            if render:
                env.render()
            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Eval episode {episode:2d}: reward = {total_reward:.0f} (survived {step + 1} steps)")

    env.close()

    print(f"\nAverage reward over {n_episodes} episodes: {np.mean(episode_rewards):.1f}")
    print(f"Min: {np.min(episode_rewards):.0f}  Max: {np.max(episode_rewards):.0f}")
    return episode_rewards


if __name__ == "__main__":
    args = init()
    evaluate(checkpoint_path=args.checkpoint, n_episodes=args.episodes, render=args.render)