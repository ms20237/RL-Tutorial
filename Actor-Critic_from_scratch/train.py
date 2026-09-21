"""
Compare this loop's SHAPE to Project 5's train.py: there, we collected
log_probs and rewards for a WHOLE episode, then called agent.update()
once at the end. Here, agent.learn() is called after EVERY step, using
only that one transition (s, log_prob, r, s', done) -- no waiting for
the episode to finish, no storing a trajectory at all.
"""
import csv
import os

import numpy as np

from environment import SimpleEnv
from agent import ActorCriticAgent, one_hot_encoder

RESULT_PATH = "./Actor-Critic_from_scratch/results"
CHECKPOINT_PATH = "./Actor-Critic_from_scratch/checkpoints"


def train(n_episodes=500, 
          max_steps=50, 
          gamma=0.99, 
          actor_lr=1e-3, 
          critic_lr=1e-2,
          seed=0, 
          results_dir=RESULT_PATH, 
          checkpoints_dir=CHECKPOINT_PATH, 
          verbose_every=50):
    np.random.seed(seed)

    env = SimpleEnv()
    encoder = one_hot_encoder(env.n_states)
    agent = ActorCriticAgent(
        state_size=env.n_states, action_size=env.n_actions, state_encoder=encoder,
        gamma=gamma, actor_lr=actor_lr, critic_lr=critic_lr,
    )

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        actor_losses, critic_losses = [], []

        for step in range(max_steps):
            action, log_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)

            actor_loss, critic_loss = agent.learn(state, log_prob, reward, next_state, done)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            state = next_state
            total_reward += reward
            if done:
                break

        history.append({
            "episode": episode, "reward": total_reward, "length": step + 1,
            "actor_loss": float(np.mean(actor_losses)), "critic_loss": float(np.mean(critic_losses)),
        })

        if episode % verbose_every == 0:
            recent = history[-verbose_every:]
            avg_reward = np.mean([h["reward"] for h in recent])
            avg_len = np.mean([h["length"] for h in recent])
            print(f"Episode {episode:4d} | avg reward (last {verbose_every}): {avg_reward:6.2f} | avg length: {avg_len:5.1f}")

    csv_path = os.path.join(results_dir, "rewards.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "actor_loss", "critic_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved training history to {csv_path}")

    ckpt_path = os.path.join(checkpoints_dir, "actor_critic_simpleenv.pt")
    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return env, agent, history


if __name__ == "__main__":
    train()