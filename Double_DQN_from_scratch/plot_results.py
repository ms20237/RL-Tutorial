"""
Reads results/rewards.csv (written by train.py) and produces the plots
section 12 of the doc asks for: reward, episode length, epsilon decay,
and training loss -- all against episode number, plus a moving average
of the reward so the trend is visible through the noise.
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "./Double_DQN_from_scratch/results/rewards.csv"
OUTPUT_PATH = "./Double_DQN_from_scratch/results"


def load_history(csv_path=CSV_PATH):
    episodes, rewards, lengths, epsilons, losses = [], [], [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            lengths.append(float(row["length"]))
            epsilons.append(float(row["epsilon"]))
            losses.append(float(row["avg_loss"]))
    return {
        "episode": np.array(episodes),
        "reward": np.array(rewards),
        "length": np.array(lengths),
        "epsilon": np.array(epsilons),
        "avg_loss": np.array(losses),
    }


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_all(history, out_dir=OUTPUT_PATH, window=20):
    os.makedirs(out_dir, exist_ok=True)

    # Reward plot (raw + moving average) 
    plt.figure(figsize=(8, 4))
    plt.plot(history["episode"], history["reward"], alpha=0.3, label="Episode reward")
    avg = moving_average(history["reward"], window)
    plt.plot(history["episode"][window - 1:window - 1 + len(avg)], avg, label=f"{window}-episode average")
    plt.axhline(475, color="green", linestyle="--", alpha=0.5, label="CartPole-v1 'solved' threshold")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title("DQN on CartPole-v1: reward over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rewards.png"))
    plt.close()

    # Episode length 
    plt.figure(figsize=(8, 4))
    plt.plot(history["episode"], history["length"], alpha=0.5)
    plt.xlabel("Episode")
    plt.ylabel("Episode length (steps survived)")
    plt.title("DQN on CartPole-v1: episode length")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "lengths.png"))
    plt.close()

    # Epsilon decay 
    plt.figure(figsize=(8, 4))
    plt.plot(history["episode"], history["epsilon"])
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration rate decay")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "epsilon.png"))
    plt.close()

    # Loss 
    plt.figure(figsize=(8, 4))
    valid = ~np.isnan(history["avg_loss"])
    plt.plot(history["episode"][valid], history["avg_loss"][valid], alpha=0.6)
    plt.xlabel("Episode")
    plt.ylabel("Average Huber loss")
    plt.title("Training loss per episode")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    print(f"Saved rewards.png, lengths.png, epsilon.png, loss.png to {out_dir}/")


if __name__ == "__main__":
    history = load_history()
    plot_all(history)