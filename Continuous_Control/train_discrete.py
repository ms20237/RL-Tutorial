"""
Trains DiscreteACAgent on DiscretizedPointMassEnv -- action menu {-1, 0, +1}.
This is the "discretize first" version the roadmap suggests building
before going fully continuous. Same online, per-step Actor-Critic update
as Project 6.
"""
import csv
import os

import numpy as np

from environment import DiscretizedPointMassEnv
from agent import DiscreteACAgent

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

    env = DiscretizedPointMassEnv(seed=seed)
    agent = DiscreteACAgent(state_size=env.n_states, n_actions=env.n_actions, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for step in range(env.max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, log_prob, reward, next_state, done)

            state = next_state
            total_reward += reward
            if done:
                break

        history.append({"episode": episode, "reward": total_reward, "length": step + 1, "final_x": float(state[0])})

        if episode % verbose_every == 0:
            recent = history[-verbose_every:]
            avg_reward = np.mean([h["reward"] for h in recent])
            avg_final_x = np.mean([abs(h["final_x"]) for h in recent])
            print(f"Episode {episode:4d} | avg reward (last {verbose_every}): {avg_reward:7.2f} | avg |final x|: {avg_final_x:.3f}")

    csv_path = os.path.join(results_dir, "rewards_discrete.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "final_x"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved training history to {csv_path}")

    ckpt_path = os.path.join(checkpoints_dir, "discrete_ac_pointmass.pt")
    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return env, agent, history


if __name__ == "__main__":
    train()