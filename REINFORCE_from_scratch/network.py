"""
Version 1 of the roadmap: "state -> neural network -> Q-values", verified
in isolation before it's wired into anything else.

This literally replaces the Q-table from Projects 1-2:

    Project 1/2:  Q(s, a)        -- a lookup in a table, self.Q[state, action]
    Project 3:    Q(s, a; theta) -- a forward pass through a neural network

Architecture (exactly as specified in the roadmap):

    Input  (4)                  cart position, velocity, pole angle, angular vel.
      |
    Linear(4 -> 64)
      |
    ReLU
      |
    Linear(64 -> 64)
      |
    ReLU
      |
    Linear(64 -> 2)             one output per action
      |
    Q(s, left), Q(s, right)

Feeding it a batch of states of shape (batch_size, 4) gives you back a
batch of Q-values of shape (batch_size, 2) -- one row per state, one
column per action. That batch dimension is what makes experience replay
possible later: instead of training on one experience at a time, we run a
whole batch of states through the network at once.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(self, n_states=4, n_actions=2, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state):
        """state: tensor of shape (batch_size, n_states) -> (batch_size, n_actions)."""
        return self.net(state)


if __name__ == "__main__":
    # Version 1 smoke test: does a forward pass produce the right shape?
    net = QNetwork(n_states=4, n_actions=2)

    single_state = torch.randn(1, 4)
    q_values = net(single_state)
    print("Single state Q-values:", q_values, "shape:", q_values.shape)

    batch_states = torch.randn(32, 4)
    batch_q_values = net(batch_states)
    print("Batch Q-values shape:", batch_q_values.shape)  # expect (32, 2)