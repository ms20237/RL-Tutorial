"""
The bridge from your control-engineering background into DRL, exactly as
the roadmap describes:

    Control Theory -> State-space model -> RL environment -> DRL controller

### The system: a 1D point mass (a "double integrator")

    x_dot = v
    v_dot = u

State:  s = [x, v]           (position, velocity)
Action: u in [-1, 1]         (a force / acceleration command)
Goal:   drive x -> 0 (and ideally v -> 0 too), starting from some x0 != 0.

This is one of the simplest possible continuous control problems, and
it's a completely standard state-space model in control theory: a
double integrator with bounded input. If you've done any state feedback
or LQR before, this reward is written to look exactly like an LQR cost:

    reward_t = -(x_t^2 + 0.01 * u_t^2)

i.e. minimize position error, with a small penalty on control effort so
the controller doesn't slam u to the limit unnecessarily. There's no
sparse "goal reached" bonus -- this is a REGULATION task (drive a
continuously-measured error to zero and hold it there), not a
navigation task like the earlier GridWorlds.

### Discretized vs continuous action, in one environment

`step(u)` always accepts a float and clips it to [-1, 1] -- the
environment itself doesn't care whether that float came from a
continuous policy or was chosen from a fixed menu of {-1, 0, +1}.
`DiscretizedPointMassEnv` below is a thin wrapper that exposes exactly
that fixed 3-action menu, so the SAME underlying dynamics can be
controlled by a discrete-action agent (Project 6's ActorCriticAgent,
unmodified) or a genuinely continuous-action agent (this project's new
ContinuousActorCriticAgent).
"""
import numpy as np


class PointMassEnv:
    def __init__(self, dt=0.1, max_steps=100, x_limit=10.0, x0_range=(-3.0, 3.0), seed=None):
        self.dt = dt
        self.max_steps = max_steps
        self.x_limit = x_limit
        self.x0_range = x0_range
        self.n_states = 2   # [x, v]
        self.rng = np.random.default_rng(seed)

        self.x = 0.0
        self.v = 0.0
        self.t = 0

    def reset(self, x0=None):
        self.x = float(self.rng.uniform(*self.x0_range)) if x0 is None else float(x0)
        self.v = 0.0
        self.t = 0
        return np.array([self.x, self.v], dtype=np.float32)

    def step(self, u):
        u = float(np.clip(u, -1.0, 1.0))

        # Euler-integrate the double integrator: x_dot = v, v_dot = u
        self.x = self.x + self.v * self.dt
        self.v = self.v + u * self.dt
        self.t += 1

        reward = -(self.x ** 2 + 0.01 * u ** 2)

        diverged = abs(self.x) > self.x_limit
        if diverged:
            reward -= 50.0  # sharp penalty for flying off -- discourages ever getting here

        timed_out = self.t >= self.max_steps
        done = diverged or timed_out

        next_state = np.array([self.x, self.v], dtype=np.float32)
        return next_state, reward, done

    def render(self, width=41):
        """Print a text 'ruler' with the mass's position marked, x in [-x_limit, x_limit]."""
        pos = int(round((self.x + self.x_limit) / (2 * self.x_limit) * (width - 1)))
        pos = max(0, min(width - 1, pos))
        center = width // 2
        ruler = ["-"] * width
        ruler[center] = "|"  # target x = 0
        ruler[pos] = "A"
        print("".join(ruler) + f"   x={self.x:+.2f}  v={self.v:+.2f}")


class DiscretizedPointMassEnv(PointMassEnv):
    """
    Same dynamics, but step() takes an integer action from {0, 1, 2},
    mapped to u in {-1, 0, +1} -- the "discretize first" version the
    roadmap suggests building before going fully continuous.
    """

    ACTION_TO_U = {0: -1.0, 1: 0.0, 2: 1.0}
    n_actions = 3

    def step(self, action):
        u = self.ACTION_TO_U[int(action)]
        return super().step(u)


if __name__ == "__main__":
    print("Continuous action smoke test (a simple hand-designed proportional controller):")
    env = PointMassEnv(seed=0)
    state = env.reset(x0=3.0)
    env.render()
    for _ in range(20):
        x, v = state
        u = -1.0 * x - 0.5 * v  # a quick PD-style action, just to see the dynamics move
        state, reward, done = env.step(u)
        env.render()
        if done:
            break

    print("\nDiscretized action smoke test:")
    denv = DiscretizedPointMassEnv(seed=0)
    state = denv.reset(x0=3.0)
    denv.render()
    for _ in range(10):
        action = 0 if state[0] > 0 else 2  # push back toward 0
        state, reward, done = denv.step(action)
        denv.render()
        if done:
            break