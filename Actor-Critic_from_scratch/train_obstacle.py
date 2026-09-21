"""
The real test for this project: Project 5's README documented REINFORCE
getting PERMANENTLY STUCK on this exact environment -- by episode ~300
it had collapsed to a deterministic bad policy (walk into the obstacle
every time) and never recovered, because it had no way to notice it was
overconfident given how little it had actually explored.

Does Actor-Critic's lower-variance, per-step advantage signal avoid that
collapse? Let's find out for real rather than assume.
"""
import csv
import os

import numpy as np

from environment import ObstacleEnv
from agent import ActorCriticAgent, one_hot_encoder

RESULT_PATH = "./Actor-Critic_from_scratch/results"
CHECKPOINT_PATH = "./Actor-Critic_from_scratch/checkpoints"


def train(n_episodes=2000, 
          max_steps=30, 
          gamma=0.99, 
          actor_lr=1e-3, 
          critic_lr=1e-2,
          seed=0, 
          results_dir=RESULT_PATH, 
          checkpoints_dir=CHECKPOINT_PATH, 
          verbose_every=100):
    np.random.seed(seed)

    env = ObstacleEnv()
    encoder = one_hot_encoder(env.n_states)
    agent = ActorCriticAgent(state_size=env.n_states, action_size=env.n_actions, state_encoder=encoder, gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, log_prob, reward, next_state, done)

            state = next_state
            total_reward += reward
            if done:
                break

        reached_goal = state == env.goal
        history.append({"episode": episode, "reward": total_reward, "length": step + 1, "reached_goal": int(reached_goal)})

        if episode % verbose_every == 0:
            recent = history[-verbose_every:]
            avg_reward = np.mean([h["reward"] for h in recent])
            success_rate = np.mean([h["reached_goal"] for h in recent])
            print(f"Episode {episode:4d} | avg reward (last {verbose_every}): {avg_reward:6.2f} | success rate: {success_rate:.0%}")

    csv_path = os.path.join(results_dir, "rewards_obstacle.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "reached_goal"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved training history to {csv_path}")

    ckpt_path = os.path.join(checkpoints_dir, "actor_critic_obstacleenv.pt")
    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return env, agent, history


if __name__ == "__main__":
    train()