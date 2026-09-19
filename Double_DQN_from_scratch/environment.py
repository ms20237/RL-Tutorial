"""
Projects 1-2 used a hand-written GridWorld to force you to understand the
environment interface from the inside. For DQN we switch to Gymnasium's
CartPole-v1 -- the classic first DRL benchmark -- but wrap it in a thin
class with the SAME reset()/step() shape you already know, so the rest of
the code (agent, train loop) looks structurally familiar.

CartPole:

              pole
               |
               |
    -----------o----------------
             cart

Actions:
    0 -> push cart left
    1 -> push cart right

State (4 continuous numbers):
    [cart position, cart velocity, pole angle, pole angular velocity]

Episode ends when the pole falls past a threshold angle, the cart leaves
the track, or the episode hits the time limit. Reward is +1 for every
timestep the pole stays up, so "reward" and "how long you survived" are
the same number.

This is exactly why a Q-table stops making sense here: the state is four
continuous numbers, so there's no way to enumerate "every state" the way
GridWorld's 12 cells could be enumerated.
"""

import gymnasium as gym


class CartPoleEnv:
    def __init__(self, render=False):
        self.env = gym.make("CartPole-v1", render_mode="human" if render else None)
        self.n_states = self.env.observation_space.shape[0]   # 4
        self.n_actions = self.env.action_space.n               # 2

    def reset(self):
        state, _info = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, terminated, truncated, _info = self.env.step(action)
        done = terminated or truncated
        return next_state, reward, done

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    # Quick smoke test: random actions for a few steps.
    env = CartPoleEnv()
    state = env.reset()
    print("Initial state:", state, "shape:", state.shape)

    for _ in range(5):
        action = env.env.action_space.sample()
        next_state, reward, done = env.step(action)
        print(f"action={action} -> state={next_state}, reward={reward}, done={done}")
        if done:
            break
    env.close()