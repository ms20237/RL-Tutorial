"""
Runs all three controllers -- discrete-action Actor-Critic, continuous-
action Actor-Critic, and the hand-designed PD baseline -- from the SAME
starting positions and plots their trajectories on the same axes. This
is the real payoff of the whole project: not "does RL beat classical
control" (on a problem this simple, well-tuned PD is essentially
optimal by construction), but "how close does a learned controller get,
and what does the discretization cost you along the way?"
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from environment import PointMassEnv, DiscretizedPointMassEnv
from agent import DiscreteACAgent, ContinuousACAgent
from pd_controller import PDController

DISCRETED_AC_POINTMASS_PATH = "./Continuous_Control/checkpoints/discrete_ac_pointmass.pt"
COUNTINUOUS_AC_POINTMASS_PATH = "./Continuous_Control/checkpoints/continuous_ac_pointmass.pt"
OUTPUT_PATH = "./Continuous_Control/results"


def run_pd(env, controller, x0):
    state = env.reset(x0=x0)
    xs, us = [state[0]], []
    total_reward = 0.0
    done = False
    while not done:
        u = controller.act(state)
        state, reward, done = env.step(u)
        xs.append(state[0])
        us.append(u)
        total_reward += reward
    return xs, us, total_reward


def run_discrete_ac(env, agent, x0):
    state = env.reset(x0=x0)
    xs, us = [state[0]], []
    total_reward = 0.0
    done = False
    while not done:
        action, _ = agent.select_action(state)
        u = env.ACTION_TO_U[action]
        state, reward, done = env.step(action)
        xs.append(state[0])
        us.append(u)
        total_reward += reward
    return xs, us, total_reward


def run_continuous_ac(env, agent, x0):
    state = env.reset(x0=x0)
    xs, us = [state[0]], []
    total_reward = 0.0
    done = False
    while not done:
        u, _ = agent.select_action(state)
        state, reward, done = env.step(u)
        xs.append(state[0])
        us.append(u)
        total_reward += reward
    return xs, us, total_reward


if __name__ == "__main__":
    x0 = 3.0

    pd_env = PointMassEnv(seed=0)
    pd_controller = PDController(kp=1.0, kd=1.5)
    pd_xs, pd_us, pd_reward = run_pd(pd_env, pd_controller, x0)

    d_env = DiscretizedPointMassEnv(seed=0)
    d_agent = DiscreteACAgent(state_size=d_env.n_states, n_actions=d_env.n_actions)
    d_agent.load(DISCRETED_AC_POINTMASS_PATH)
    d_xs, d_us, d_reward = run_discrete_ac(d_env, d_agent, x0)

    c_env = PointMassEnv(seed=0)
    c_agent = ContinuousACAgent(state_size=c_env.n_states, action_size=1)
    c_agent.load(COUNTINUOUS_AC_POINTMASS_PATH)
    c_agent.std = 1e-6  # evaluate ~deterministically: negligible exploration noise
    c_xs, c_us, c_reward = run_continuous_ac(c_env, c_agent, x0)

    print(f"Starting at x0={x0}")
    print(f"  PD controller:         total reward = {pd_reward:7.2f}, final x = {pd_xs[-1]:+.3f}, steps = {len(pd_xs)-1}")
    print(f"  Discrete AC (3 acts):  total reward = {d_reward:7.2f}, final x = {d_xs[-1]:+.3f}, steps = {len(d_xs)-1}")
    print(f"  Continuous AC:         total reward = {c_reward:7.2f}, final x = {c_xs[-1]:+.3f}, steps = {len(c_xs)-1}")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax1.plot(pd_xs, label=f"PD controller (reward={pd_reward:.1f})")
    ax1.plot(d_xs, label=f"Discrete AC (reward={d_reward:.1f})")
    ax1.plot(c_xs, label=f"Continuous AC (reward={c_reward:.1f})")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Position x")
    ax1.set_title(f"Trajectories from x0={x0}: PD vs Discrete AC vs Continuous AC")
    ax1.legend()

    ax2.step(range(len(pd_us)), pd_us, where="post", label="PD controller", alpha=0.8)
    ax2.step(range(len(d_us)), d_us, where="post", label="Discrete AC", alpha=0.8)
    ax2.step(range(len(c_us)), c_us, where="post", label="Continuous AC", alpha=0.8)
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Control action u")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/comparison_trajectories.png")
    print(f"\nSaved {OUTPUT_PATH}/comparison_trajectories.png")