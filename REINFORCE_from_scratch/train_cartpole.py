"""
Section 17: once the tiny world works, move to CartPole. As the doc
says, "the rest of your algorithm remains essentially the same" -- and
here that's not just a claim, it's literally true in this code: this
file imports the exact same REINFORCEAgent class from agent.py that
train.py and train_obstacle.py use. The only things that change are:

    - the environment (CartPole instead of a hand-written world)
    - the state encoder (identity -- CartPole's state is already 4 floats,
      there's nothing to one-hot encode)
    - the policy network's input size (4 instead of 5 or 12)

Requires: pip install "gymnasium[classic-control]"
"""

import csv
import os

import gymnasium as gym
import numpy as np

from agent import REINFORCEAgent, identity_encoder

RESULT_PATH = "./REINFORCE_from_scratch/results"
CHECKPOINT_PATH = "./REINFORCE_from_scratch/checkpoints"


def train(n_episodes=1000, 
          max_steps=500, 
          gamma=0.99, 
          lr=1e-2, 
          normalize_returns=True,
          seed=0, 
          results_dir=RESULT_PATH, 
          checkpoints_dir=CHECKPOINT_PATH, 
          verbose_every=25):
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]  # 4
    n_actions = env.action_space.n              # 2

    agent = REINFORCEAgent(
        state_size=n_states,
        action_size=n_actions,
        state_encoder=identity_encoder,   # <-- the only real change from train.py
        hidden_size=64,                   # a bit more capacity than the toy world's 32
        gamma=gamma,
        lr=lr,
        normalize_returns=normalize_returns,
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = []

    for episode in range(1, n_episodes + 1):
        state, _info = env.reset()
        log_probs, rewards = [], []

        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state
            if done:
                break

        loss = agent.update(log_probs, rewards)

        total_reward = sum(rewards)
        history.append({"episode": episode, "reward": total_reward, "length": step + 1, "loss": loss})

        if episode % verbose_every == 0:
            recent = history[-verbose_every:]
            avg_reward = np.mean([h["reward"] for h in recent])
            print(f"Episode {episode:4d} | avg reward (last {verbose_every}): {avg_reward:6.1f}")

    env.close()

    csv_path = os.path.join(results_dir, "rewards_cartpole.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved training history to {csv_path}")

    ckpt_path = os.path.join(checkpoints_dir, "reinforce_cartpole.pt")
    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return agent, history


if __name__ == "__main__":
    train()