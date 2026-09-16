"""
Two things worth seeing after training:
1. The reward-per-episode curve (did the agent actually learn?)
2. The greedy path the trained agent now takes from S to G.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from environment import GridWorld
from train import train

PLOT_PATH = "./Q-Learning_from_scratch/plots"


def plot_rewards(episode_rewards, window=20, plot_path=PLOT_PATH):
    """Plot raw rewards plus a rolling average to smooth out the noise."""
    rewards = np.array(episode_rewards)
    if len(rewards) >= window:
        rolling_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
    else:
        rolling_avg = rewards

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, alpha=0.3, label="Episode reward")
    plt.plot(range(window - 1, window - 1 + len(rolling_avg)), rolling_avg, label=f"{window}-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("Q-Learning on GridWorld: reward over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_path}/reward_curve.png")
    print(f"Saved {plot_path}/reward_curve.png")


def print_greedy_path(env, agent, max_steps=20):
    """Run the agent greedily (no exploration) and print the path it takes."""
    state = env.reset()
    path = [env._to_pos(state)]

    print("\nGreedy path from S to G:")
    env.render()

    for _ in range(max_steps):
        action = int(np.argmax(agent.Q[state]))
        next_state, reward, done = env.step(action)
        path.append(env._to_pos(next_state))
        print(f"  -> {env.action_names[action]} -> {env._to_pos(next_state)} (reward {reward})")

        state = next_state
        if done:
            break

    env.render()
    print("Full path (row, col):", path)


def print_policy_grid(env, agent):
    """Print an arrow for the best action in every non-obstacle cell."""
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
    print("\nLearned policy (best action per cell):")
    for r in range(env.n_rows):
        row_str = []
        for c in range(env.n_cols):
            pos = (r, c)
            if pos in env.obstacles:
                row_str.append(" X ")
            elif pos == env.goal:
                row_str.append(" G ")
            else:
                state = r * env.n_cols + c
                best_action = int(np.argmax(agent.Q[state]))
                row_str.append(f" {arrows[best_action]} ")
        print("|".join(row_str))


if __name__ == "__main__":
    env, agent, rewards, lengths = train(n_episodes=500, verbose_every=100)

    os.makedirs(PLOT_PATH, exist_ok=True)
    plot_rewards(rewards, plot_path=PLOT_PATH)
    print_policy_grid(env, agent)
    print_greedy_path(env, agent)