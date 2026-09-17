"""
The SARSA training loop.
Q-learning's loop (off-policy):

    state = env.reset()
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update(state, action, reward, next_state)      # only needs s,a,r,s'
        state = next_state

SARSA's loop (on-policy):

    state = env.reset()
    action = agent.choose_action(state)                       # chosen ONCE, up front
    while not done:
        next_state, reward, done = env.step(action)
        next_action = agent.choose_action(next_state)         # chosen BEFORE the update
        agent.update(state, action, reward, next_state, next_action)  # needs s,a,r,s',a'
        state, action = next_state, next_action

This is the concrete difference between "on-policy" and "off-policy":
SARSA's update depends on which action the agent's own (epsilon-greedy)
policy actually decided to take next -- including any exploratory random
moves. Q-learning's update ignores that and always assumes the best
possible next action.
"""
import os
import numpy as np

from environment import GridWorld
from agent import SarsaAgent

SAVE_MODEL_PATH = "./SARSA_from_scratch/models"


def train(n_episodes=500, max_steps=100, verbose_every=50, seed=0):
    np.random.seed(seed)

    env = GridWorld()
    agent = SarsaAgent(n_states=env.n_states, n_actions=env.n_actions)

    episode_rewards = []
    episode_lengths = []

    for episode in range(1, n_episodes + 1):
        state = env.reset()
        action = agent.choose_action(state)  # pick the first action up front

        total_reward = 0
        done = False

        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent.choose_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            state, action = next_state, next_action
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
    np.save(f"{SAVE_MODEL_PATH}/q_table_sarsa.npy", agent.Q)
    print(f"\nSaved Q-table to {SAVE_MODEL_PATH}/q_table_sarsa.npy")