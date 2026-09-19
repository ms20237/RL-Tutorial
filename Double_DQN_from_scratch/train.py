"""
Identical training loop to Project 3, just importing DoubleDQNAgent
instead of DQNAgent. That's the whole point: Double DQN is a drop-in
replacement, not a different algorithm from the outside.
    1. reset environment
    2. select action (epsilon-greedy)
    3. execute action
    4. save experience to replay memory
    5. sample a random batch
    6. compute current Q-values          }
    7. compute Bellman target            } all inside agent.learn()
    8. compute loss                      }
    9. backpropagate                     }
   10. occasionally sync target network  }

We don't start training until the replay buffer has at least
`min_buffer_size` experiences in it -- training on a handful of samples
sampled-with-replacement from a nearly-empty buffer defeats the purpose
of decorrelating experiences.

We log, per episode: total reward, episode length, epsilon, and the
average training loss -- exactly the metrics section 12 of the doc calls
out, so you can see not just "did it solve CartPole" but "how".
"""

import csv
import os

import numpy as np

from environment import CartPoleEnv
from agent import DoubleDQNAgent
from replay_buffer import ReplayBuffer

RESULT_PATH = "./Double_DQN_from_scratch/results"
CHECKPOINT_PATH = "./Double_DQN_from_scratch/checkpoints"


def train(
    n_episodes=400,
    max_steps=500,
    batch_size=64,
    min_buffer_size=1000,
    buffer_capacity=10_000,
    seed=0,
    results_dir=RESULT_PATH,
    checkpoints_dir=CHECKPOINT_PATH,
    verbose_every=10,
):
    np.random.seed(seed)

    env = CartPoleEnv()
    agent = DoubleDQNAgent(n_states=env.n_states, n_actions=env.n_actions)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    history = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0.0
        losses = []

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if len(buffer) >= min_buffer_size:
                batch = buffer.sample(batch_size)
                loss = agent.learn(batch)
                losses.append(loss)

            if done:
                break

        agent.decay_epsilon()

        avg_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(
            {
                "episode": episode,
                "reward": total_reward,
                "length": step + 1,
                "epsilon": agent.epsilon,
                "avg_loss": avg_loss,
            }
        )

        if episode % verbose_every == 0:
            recent = history[-verbose_every:]
            avg_reward = np.mean([h["reward"] for h in recent])
            avg_len = np.mean([h["length"] for h in recent])
            print(
                f"Episode {episode:4d} | "
                f"avg reward (last {verbose_every}): {avg_reward:6.1f} | "
                f"avg length: {avg_len:6.1f} | "
                f"epsilon: {agent.epsilon:.3f} | "
                f"avg loss: {avg_loss:.4f}"
            )

    env.close()

    csv_path = os.path.join(results_dir, "rewards.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "epsilon", "avg_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nSaved training history to {csv_path}")

    ckpt_path = os.path.join(checkpoints_dir, "double_dqn_cartpole.pt")
    agent.save(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    return agent, history


if __name__ == "__main__":
    train()