"""
The training loop. This is the piece you should be able to explain
line by line before moving on to Project 2 (SARSA).

Conceptually:

    initialize Q-table

    for episode:
        state = environment.reset()
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = environment.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
"""
import os
import numpy as np

from environment import GridWorld
from agent import QLearningAgent


SAVE_MODEL_PATH = "./Q-Learning_from_scratch/models"


def train(n_episodes=500, max_steps=100, verbose_every=50, seed=0):
    np.random.seed(seed)

    env = GridWorld()
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions)

    episode_rewards = []
    episode_lengths = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()

        episode_rewards.append(total_reward)
        episode_lengths.append(step + 1)

        if episode % verbose_every == 0:
            avg_reward = np.mean(episode_rewards[-verbose_every:])
            print(
                f"Episode {episode:4d} | "
                f"avg reward (last {verbose_every}): {avg_reward:6.2f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )

    return env, agent, episode_rewards, episode_lengths


if __name__ == "__main__":
    env, agent, rewards, lengths = train(n_episodes=500)

    print("\nTraining finished.")
    print("Learned Q-table (rows = states, cols = actions [UP, RIGHT, DOWN, LEFT]):")
    print(np.round(agent.Q, 2))

    # Save the Q-table so visualize.py can load it separately if you want.
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)
    np.save(f"{SAVE_MODEL_PATH}/q_table.npy", agent.Q)
    print(f"\nSaved Q-table to {SAVE_MODEL_PATH}/q_table.npy")