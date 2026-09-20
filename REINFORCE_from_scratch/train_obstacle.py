"""
make the environment slightly harder before jumping to CartPole. The obstacle forces the agent to learn something
less trivial than "always go RIGHT" -- it has to learn to sometimes go
LEFT first, or the run ends immediately in a -10 penalty.

    S . . X . . G
    0 1 2 3 4 5 6

Structurally this file is almost identical to train.py -- same
REINFORCEAgent, same update() -- only the environment changes. That's
the point: the algorithm doesn't need to know anything about *why* the
world is harder.
"""

import csv
import os

import numpy as np

from environment import ObstacleEnv
from agent import REINFORCEAgent, one_hot_encoder

RESULT_PATH = "./REINFORCE_from_scratch/results"
CHECKPOINT_PATH = "./REINFORCE_from_scratch/checkpoints"


def train(n_episodes=2000, 
          max_steps=30, 
          gamma=0.99, lr=1e-2, 
          normalize_returns=True,
          seed=0, 
          results_dir=RESULT_PATH, 
          checkpoints_dir=CHECKPOINT_PATH, 
          verbose_every=100):
    """
    normalize_returns defaults to True here (unlike train.py's default of
    False) -- with an obstacle nearby, raw returns swing much more wildly
    episode to episode (-10 vs +10 vs everything in between), and that
    extra variance makes plain REINFORCE slower and noisier to converge.
    Try normalize_returns=False yourself and compare -- it's a good way
    to feel section 18's "training can be very noisy" problem directly.
    """
    np.random.seed(seed)

    env = ObstacleEnv()
    encoder = one_hot_encoder(env.n_states)
    agent = REINFORCEAgent(
        state_size=env.n_states,
        action_size=env.n_actions,
        state_encoder=encoder,
        gamma=gamma,
        lr=lr,
        normalize_returns=normalize_returns,
    )

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
        reached_goal = state == env.goal
        history.append({
            "episode": episode, "reward": total_reward, "length": step + 1,
            "loss": loss, "reached_goal": int(reached_goal),
        })

        if episode % verbose_every == 0:
            recent = history[-verbose_every:]
            avg_reward = np.mean([h["reward"] for h in recent])
            success_rate = np.mean([h["reached_goal"] for h in recent])
            print(f"Episode {episode:4d} | avg reward (last {verbose_every}): {avg_reward:6.2f} | success rate: {success_rate:.0%}")

    csv_path = os.path.join(results_dir, "rewards_obstacle.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "loss", "reached_goal"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved training history to {csv_path}")

    ckpt_path = os.path.join(checkpoints_dir, "reinforce_obstacleenv.pt")
    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return env, agent, history


if __name__ == "__main__":
    train()