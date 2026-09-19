"""
Why this exists: consecutive experiences (s1,a1,r1,s2) -> (s2,a2,r2,s3) ->
(s3,a3,r3,s4) are highly correlated -- s2 depends directly on s1, etc.
Neural networks trained on a stream of correlated samples tend to
overfit to whatever situation the agent is currently in and forget
what it learned a few steps ago.

Experience replay fixes this by storing every transition in a big buffer
and training on a random batch sampled from the WHOLE history each step,
so consecutive training batches are no longer temporally correlated.

Storage:

    Replay Buffer
    +----------------------------+
    | (s1, a1, r1, s2, done)     |
    | (s7, a0, r7, s8, done)     |
    | (s3, a1, r3, s4, done)     |
    | ...                        |
    +----------------------------+

The buffer has a fixed maximum size (`capacity`). Once full, the oldest
experiences are dropped to make room for new ones (a "circular buffer"),
using Python's collections.deque(maxlen=...) which does this for free.
"""

import random
from collections import deque, namedtuple

import numpy as np

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Return a random batch as separate numpy arrays, ready for torch.tensor()."""
        batch = random.sample(self.buffer, batch_size)

        states = np.array([t.state for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_states = np.array([t.next_state for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    # Smoke test: push some fake transitions, sample a batch, check shapes.
    buf = ReplayBuffer(capacity=100)
    for i in range(50):
        state = np.random.randn(4).astype(np.float32)
        next_state = np.random.randn(4).astype(np.float32)
        buf.push(state, action=i % 2, reward=1.0, next_state=next_state, done=False)

    print("Buffer size:", len(buf))

    states, actions, rewards, next_states, dones = buf.sample(batch_size=8)
    print("states:", states.shape)       # (8, 4)
    print("actions:", actions.shape)     # (8,)
    print("rewards:", rewards.shape)     # (8,)
    print("next_states:", next_states.shape)  # (8, 4)
    print("dones:", dones.shape)         # (8,)