"""
Same idea as Project 5's evaluate.py: step through a trained policy on
SimpleEnv, printing the ASCII grid and the actor's action probabilities
at each state, plus the critic's value estimate V(s) -- new in this
project, and worth watching: it should roughly track "how many steps of
-1 until the +10" from each state.
"""
import argparse

import torch

from environment import SimpleEnv
from agent import ActorCriticAgent, one_hot_encoder

CHECKPOINT_PATH = "./Actor-Critic_from_scratch/checkpoints/actor_critic_simpleenv.pt"


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--greedy", action="store_true", help="Use argmax instead of sampling")

    args = parser.parse_args()
    return args


def evaluate(checkpoint_path=CHECKPOINT_PATH, n_episodes=2, max_steps=20, stochastic=True):
    env = SimpleEnv()
    encoder = one_hot_encoder(env.n_states)
    agent = ActorCriticAgent(state_size=env.n_states, action_size=env.n_actions, state_encoder=encoder)
    agent.load(checkpoint_path)

    action_names = {0: "LEFT", 1: "RIGHT"}

    for episode in range(1, n_episodes + 1):
        print(f"\n=== Episode {episode} ===")
        state = env.reset()
        env.render()
        total_reward = 0

        for step in range(max_steps):
            state_t = encoder(state).unsqueeze(0).to(agent.device)
            probs = agent.actor(state_t).detach().squeeze(0).cpu()
            value = agent.critic(state_t).detach().item()
            print(f"  state={state}  P(LEFT)={probs[0]:.2f}  P(RIGHT)={probs[1]:.2f}  V(s)={value:.2f}")

            if stochastic:
                action, _ = agent.select_action(state)
            else:
                action = int(torch.argmax(probs).item())

            next_state, reward, done = env.step(action)
            print(f"  -> {action_names[action]} -> reward {reward}")
            env.render()

            state = next_state
            total_reward += reward
            if done:
                break

        print(f"Episode {episode} total reward: {total_reward}")


if __name__ == "__main__":
    args = init()
    evaluate(checkpoint_path=args.checkpoint, n_episodes=args.episodes, stochastic=not args.greedy)