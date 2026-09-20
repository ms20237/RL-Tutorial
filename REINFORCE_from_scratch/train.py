"""
The training loop from section 13 of the doc, with logging added so we
can actually see the noisy-reward problem section 18 warns about.

    for episode:
        collect one full episode (states, actions never stored here --
                                   only log_probs and rewards, since
                                   that's all update() needs)
        compute returns
        compute policy-gradient loss
        backprop
"""

import csv
import os

import numpy as np

from environment import SimpleEnv
from agent import REINFORCEAgent, one_hot_encoder

RESULT_PATH = "./REINFORCE_from_scratch/results"
CHECKPOINT_PATH = "./REINFORCE_from_scratch/checkpoints"


def train(n_episodes=500, 
          max_steps=50, 
          gamma=0.99, 
          lr=1e-2, 
          normalize_returns=False,
          seed=0, 
          results_dir=RESULT_PATH, 
          checkpoints_dir=CHECKPOINT_PATH, 
          verbose_every=50):
    np.random.seed(seed)

    env = SimpleEnv()
    encoder = one_hot_encoder(env.n_states)
    agent = REINFORCEAgent(
        state_size=env.n_states,
        action_size=env.n_actions,
        state_encoder=encoder,
        gamma=gamma,
        lr=lr,
        normalize_returns=normalize_returns,)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        log_probs, rewards = [], []

        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)

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
            avg_len = np.mean([h["length"] for h in recent])
            print(f"Episode {episode:4d} | avg reward (last {verbose_every}): {avg_reward:6.2f} | avg length: {avg_len:5.1f}")

    csv_path = os.path.join(results_dir, "rewards.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved training history to {csv_path}")

    ckpt_path = os.path.join(checkpoints_dir, "reinforce_simpleenv.pt")
    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return env, agent, history


if __name__ == "__main__":
    train()