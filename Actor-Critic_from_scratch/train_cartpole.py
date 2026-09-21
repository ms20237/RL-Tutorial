"""
Same test as Project 5's train_cartpole.py, same ActorCriticAgent class
used for SimpleEnv and ObstacleEnv above -- only the environment and
state encoder change (identity_encoder, since CartPole's state is
already 4 floats). Project 5's REINFORCE result on CartPole repeatedly
solved it (reward 500) then collapsed back under 100, over and over.
Does the lower-variance, per-step advantage signal calm that down?
"""

import csv
import os

import gymnasium as gym
import numpy as np

from agent import ActorCriticAgent, identity_encoder

RESULT_PATH = "./Actor-Critic_from_scratch/results"
CHECKPOINT_PATH = "./Actor-Critic_from_scratch/checkpoints"


def train(n_episodes=1000, 
          max_steps=500, 
          gamma=0.99, 
          actor_lr=1e-3, 
          critic_lr=1e-2,
          seed=0, 
          results_dir=RESULT_PATH, 
          checkpoints_dir=CHECKPOINT_PATH, 
          verbose_every=25):
    np.random.seed(seed)

    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ActorCriticAgent(state_size=n_states, action_size=n_actions, state_encoder=identity_encoder, hidden_size=64, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = []

    for episode in range(1, n_episodes + 1):
        state, _info = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            agent.learn(state, log_prob, reward, next_state, done)

            state = next_state
            total_reward += reward
            if done:
                break

        history.append({"episode": episode, "reward": total_reward, "length": step + 1})

        if episode % verbose_every == 0:
            recent = history[-verbose_every:]
            avg_reward = np.mean([h["reward"] for h in recent])
            print(f"Episode {episode:4d} | avg reward (last {verbose_every}): {avg_reward:6.1f}")

    env.close()

    csv_path = os.path.join(results_dir, "rewards_cartpole.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved training history to {csv_path}")

    ckpt_path = os.path.join(checkpoints_dir, "actor_critic_cartpole.pt")
    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return agent, history


if __name__ == "__main__":
    train()