"""
Loads Project 5's REINFORCE result and this project's Actor-Critic
result on the SAME obstacle GridWorld and overlays them. This is the
most important plot in this project -- it turns "Actor-Critic reduces
variance" from a claim into something you can see directly.

Run Project 5's train_obstacle.py first if its results/rewards_obstacle.csv
doesn't exist yet.
"""
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

REINFORCE_REWARD_PATH = "./REINFORCE_from_scratch/results/rewards_obstacle.csv"
ACTOR_CRITIC_CSV_REWARD_PATH = "./Actor-Critic_from_scratch/results/rewards_obstacle.csv"
OUTPUT_PATH = "./Actor-Critic_from_scratch/results"


def load_rewards(csv_path):
    episodes, rewards = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
    return np.array(episodes), np.array(rewards)


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


if __name__ == "__main__":
    reinforce_csv = REINFORCE_REWARD_PATH
    actor_critic_csv = ACTOR_CRITIC_CSV_REWARD_PATH

    window = 20
    ep_r, reward_r = load_rewards(reinforce_csv)
    ep_ac, reward_ac = load_rewards(actor_critic_csv)

    avg_r = moving_average(reward_r, window)
    avg_ac = moving_average(reward_ac, window)

    plt.figure(figsize=(8, 4))
    plt.plot(ep_r[window - 1:window - 1 + len(avg_r)], avg_r, label="REINFORCE (Project 5)")
    plt.plot(ep_ac[window - 1:window - 1 + len(avg_ac)], avg_ac, label="Actor-Critic (this project)")
    plt.xlabel("Episode")
    plt.ylabel(f"Reward ({window}-episode rolling average)")
    plt.title("Same obstacle GridWorld: REINFORCE vs Actor-Critic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/comparison_obstacle.png")
    plt.close()

    print(f"Saved {OUTPUT_PATH}/comparison_obstacle.png")
    print(f"\nFinal 100-episode average reward -- REINFORCE: {reward_r[-100:].mean():.2f}   Actor-Critic: {reward_ac[-100:].mean():.2f}")

    # How many of the final episodes did each algorithm actually reach the goal?
    # (A reward of ~6-7 means goal reached; -10 or less means it hit the obstacle.)
    final_success_reinforce = np.mean(reward_r[-100:] > 0)
    final_success_ac = np.mean(reward_ac[-100:] > 0)
    print(f"Final 100-episode success rate -- REINFORCE: {final_success_reinforce:.0%}   Actor-Critic: {final_success_ac:.0%}")