"""
Loads the saved results/rewards.csv from BOTH Project 3 (vanilla DQN) and
this project (Double DQN) -- same environment, same hyperparameters, same
seed, same number of episodes -- and overlays their reward and loss
curves so the effect of the one-line target change is visible directly,
not just described.

Run Project 3's train.py first if you haven't already, so its
results/rewards.csv exists.
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

DQN_CSV_REWARD_PATH = "./DQN_from_Scratch/results/rewards.csv"
DOUBLE_DQN_CSV_REWARD_PATH = "./Double_DQN_from_scratch/results/rewards.csv"
OUTPUT_PATH = "./Double_DQN_from_Scratch/results"


def load_history(csv_path):
    episodes, rewards, losses = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            losses.append(float(row["avg_loss"]))
    return np.array(episodes), np.array(rewards), np.array(losses)


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


if __name__ == "__main__":
    dqn_csv = DQN_CSV_REWARD_PATH
    double_dqn_csv = DOUBLE_DQN_CSV_REWARD_PATH

    if not os.path.exists(dqn_csv):
        raise FileNotFoundError(
            f"Couldn't find {dqn_csv} -- run Project 3's train.py first so there's a "
            "vanilla-DQN result to compare against.")

    window = 20

    ep_dqn, reward_dqn, loss_dqn = load_history(dqn_csv)
    ep_ddqn, reward_ddqn, loss_ddqn = load_history(double_dqn_csv)

    avg_reward_dqn = moving_average(reward_dqn, window)
    avg_reward_ddqn = moving_average(reward_ddqn, window)

    # Reward comparison 
    plt.figure(figsize=(8, 4))
    plt.plot(ep_dqn[window - 1:window - 1 + len(avg_reward_dqn)], avg_reward_dqn, label="DQN (vanilla)")
    plt.plot(ep_ddqn[window - 1:window - 1 + len(avg_reward_ddqn)], avg_reward_ddqn, label="Double DQN")
    plt.axhline(475, color="green", linestyle="--", alpha=0.5, label="'Solved' threshold")
    plt.xlabel("Episode")
    plt.ylabel(f"Reward ({window}-episode rolling average)")
    plt.title("DQN vs Double DQN on the same CartPole-v1 setup")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/comparison_rewards.png")
    plt.close()

    # Loss comparison (the overestimation signature) 
    valid_dqn = ~np.isnan(loss_dqn)
    valid_ddqn = ~np.isnan(loss_ddqn)

    plt.figure(figsize=(8, 4))
    plt.plot(ep_dqn[valid_dqn], loss_dqn[valid_dqn], alpha=0.5, label="DQN (vanilla)")
    plt.plot(ep_ddqn[valid_ddqn], loss_ddqn[valid_ddqn], alpha=0.5, label="Double DQN")
    plt.xlabel("Episode")
    plt.ylabel("Average Huber loss")
    plt.title("Training loss: DQN vs Double DQN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/comparison_loss.png")
    plt.close()

    print(f"Saved {OUTPUT_PATH}/comparison_rewards.png and {OUTPUT_PATH}/comparison_loss.png")
    print(f"\nFinal 50-episode average reward -- DQN: {reward_dqn[-50:].mean():.1f}   Double DQN: {reward_ddqn[-50:].mean():.1f}")
    print(f"Peak single-episode reward       -- DQN: {reward_dqn.max():.0f}   Double DQN: {reward_ddqn.max():.0f}")