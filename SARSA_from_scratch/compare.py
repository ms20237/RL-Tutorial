"""
Run Q-learning and SARSA on the EXACT same GridWorld and compare:
  - final learned policy (arrows)
  - final greedy path
  - reward curves, overlaid

This is the experiment the roadmap calls out explicitly: same environment,
same hyperparameters, only the TD target differs. Whatever difference you
see in the results comes purely from on-policy vs off-policy learning.
"""
import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

from environment import GridWorld  
from agent import SarsaAgent     

COMPARE_PLOT_PATH = "./SARSA_from_scratch/plots"


def _load_class_from_file(file_path, module_name, class_name):
    """
    Load QLearningAgent from Project 1's agent.py by explicit file path,
    under its own module name, so it doesn't collide with this folder's
    local `agent` module (which also defines a class called `agent.py`).
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


_q_learning_agent_path = os.path.join(os.path.dirname(__file__), "..", "Q-Learning_from_scratch", "agent.py")
QLearningAgent = _load_class_from_file(_q_learning_agent_path, "q_learning_agent_module", "QLearningAgent")


def run(agent_cls, is_sarsa, n_episodes=500, max_steps=100, seed=0):
    np.random.seed(seed)
    env = GridWorld()
    agent = agent_cls(n_states=env.n_states, n_actions=env.n_actions)
    episode_rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        if is_sarsa:
            action = agent.choose_action(state)
            for step in range(max_steps):
                next_state, reward, done = env.step(action)
                next_action = agent.choose_action(next_state)
                agent.update(state, action, reward, next_state, next_action, done)
                state, action = next_state, next_action
                total_reward += reward
                if done:
                    break
        else:
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

    return env, agent, episode_rewards


def greedy_path(env, agent, max_steps=20):
    state = env.reset()
    path = [env._to_pos(state)]
    for _ in range(max_steps):
        action = int(np.argmax(agent.Q[state]))
        next_state, reward, done = env.step(action)
        path.append(env._to_pos(next_state))
        state = next_state
        if done:
            break
    return path


def policy_grid_str(env, agent):
    arrows = {0: "^", 1: ">", 2: "v", 3: "<"}
    lines = []
    for r in range(env.n_rows):
        row_str = []
        for c in range(env.n_cols):
            pos = (r, c)
            if pos in env.obstacles:
                row_str.append(" X ")
            elif pos == env.goal:
                row_str.append(" G ")
            else:
                state = r * env.n_cols + c
                best_action = int(np.argmax(agent.Q[state]))
                row_str.append(f" {arrows[best_action]} ")
        lines.append("|".join(row_str))
    return "\n".join(lines)


if __name__ == "__main__":
    print("Training Q-learning (off-policy)...")
    env_q, agent_q, rewards_q = run(QLearningAgent, is_sarsa=False)

    print("Training SARSA (on-policy)...")
    env_s, agent_s, rewards_s = run(SarsaAgent, is_sarsa=True)

    print("\n=== Q-learning policy ===")
    print(policy_grid_str(env_q, agent_q))
    print("Greedy path:", greedy_path(env_q, agent_q))

    print("\n=== SARSA policy ===")
    print(policy_grid_str(env_s, agent_s))
    print("Greedy path:", greedy_path(env_s, agent_s))

    # Overlay reward curves
    window = 20
    def rolling(x):
        x = np.array(x)
        return np.convolve(x, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(rolling(rewards_q), label="Q-learning (off-policy)")
    plt.plot(rolling(rewards_s), label="SARSA (on-policy)")
    plt.xlabel("Episode")
    plt.ylabel(f"Reward ({window}-episode rolling average)")
    plt.title("Q-learning vs SARSA on the same GridWorld")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{COMPARE_PLOT_PATH}/comparison.png")
    print("\nSaved comparison.png")