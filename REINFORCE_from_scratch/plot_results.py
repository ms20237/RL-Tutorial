"""
Plots reward-per-episode from results/rewards.csv. On the 1D world you
should see fast, fairly clean convergence (few states, short episodes).
When you run this against CartPole's results later, expect the section-18
noisiness the doc warns about -- much bigger swings episode to episode.
"""

import csv
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "./REINFORCE_from_scratch/results/rewards.csv"
OUTPUT_PATH = "./REINFORCE_from_scratch/results/rewards.png"

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CSV_PATH)
    parser.add_argument("--out", default=OUTPUT_PATH)
    parser.add_argument("--title", default="REINFORCE: reward over time")
    
    args = parser.parse_args()
    return args


def load_history(csv_path):
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


def plot(csv_path=CSV_PATH, out_path=OUTPUT_PATH, title="REINFORCE: reward over time", window=20):
    episodes, rewards = load_history(csv_path)
    avg = moving_average(rewards, window)

    plt.figure(figsize=(8, 4))
    plt.plot(episodes, rewards, alpha=0.3, label="Episode reward")
    plt.plot(episodes[window - 1:window - 1 + len(avg)], avg, label=f"{window}-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(f"{out_path}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    args = init()
    plot(csv_path=args.csv, out_path=args.out, title=args.title)