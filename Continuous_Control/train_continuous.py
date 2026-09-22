"""
Trains ContinuousACAgent on PointMassEnv directly -- u sampled from a
Normal distribution and clipped to [-1, 1], not chosen from a fixed
menu. Structurally almost identical to train_discrete.py -- compare the
two files side by side, the only differences are the environment class
and the agent class.
"""
import csv
import os

import numpy as np

from environment import PointMassEnv
from agent import ContinuousACAgent

RESULT_PATH = "./Continuous_Control/results"
CHECKPOINT_PATH = "./Continuous_Control/checkpoints"


def train(n_episodes=800, 
          gamma=0.99, 
          actor_lr=1e-3, 
          critic_lr=1e-2,
          seed=0, 
          results_dir=RESULT_PATH, 
          checkpoints_dir=CHECKPOINT_PATH, 
          verbose_every=50):
    np.random.seed(seed)

    env = PointMassEnv(seed=seed)
    agent = ContinuousACAgent(state_size=env.n_states, action_size=1, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for step in range(env.max_steps):
            u, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(u)
            agent.learn(state, log_prob, reward, next_state, done)

            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_std()

        history.append({"episode": episode, "reward": total_reward, "length": step + 1, "final_x": float(state[0])})

        if episode % verbose_every == 0:
            recent = history[-verbose_every:]
            avg_reward = np.mean([h["reward"] for h in recent])
            avg_final_x = np.mean([abs(h["final_x"]) for h in recent])
            print(f"Episode {episode:4d} | avg reward (last {verbose_every}): {avg_reward:7.2f} | avg |final x|: {avg_final_x:.3f} | std: {agent.std:.3f}")

    csv_path = os.path.join(results_dir, "rewards_continuous.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "final_x"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved training history to {csv_path}")

    ckpt_path = os.path.join(checkpoints_dir, "continuous_ac_pointmass.pt")
    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return env, agent, history


if __name__ == "__main__":
    train()