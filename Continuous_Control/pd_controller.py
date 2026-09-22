"""
The classical control baseline: a hand-designed PD (proportional-derivative)
controller, no learning at all. This is the natural comparison point given
a control-engineering background -- "how does a DRL controller compare to
what I'd design by hand in five minutes?"

For the double integrator (x_dot = v, v_dot = u), a PD law on position
error e = x - x_target (x_target = 0 here, so e = x):

    u = -Kp * x - Kd * v

is a completely standard, textbook state-feedback controller. With the
right gains this is actually OPTIMAL for this exact system and reward
(the LQR solution for a double integrator with a quadratic cost looks
exactly like this form) -- so it's a genuinely strong baseline, not a
strawman. Seeing how close a learned controller gets to it is the
interesting comparison, not "does RL beat classical control" (on a
problem this simple and this well-understood, classical control wins by
design).
"""
import numpy as np
from environment import PointMassEnv


class PDController:
    def __init__(self, kp=1.0, kd=1.5):
        self.kp = kp
        self.kd = kd

    def act(self, state):
        x, v = state
        u = -self.kp * x - self.kd * v
        return float(np.clip(u, -1.0, 1.0))


if __name__ == "__main__":
    env = PointMassEnv(seed=0)
    controller = PDController(kp=1.0, kd=1.5)

    state = env.reset(x0=3.0)
    env.render()
    total_reward = 0.0
    done = False
    while not done:
        u = controller.act(state)
        state, reward, done = env.step(u)
        total_reward += reward
        env.render()

    print(f"\nTotal reward: {total_reward:.2f}")