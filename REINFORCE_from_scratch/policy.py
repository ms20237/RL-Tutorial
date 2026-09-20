"""
The core conceptual shift from Projects 1-4: instead of a network that
outputs Q(s,a) for every action and then taking argmax, this network
outputs a full probability distribution over actions:

    pi_theta(a|s) = softmax(logits)

For example, given some state:

    Left   0.20
    Right  0.80

We SAMPLE from this distribution rather than always taking the max --
that's what gives the policy exploration "for free," without needing a
separate epsilon-greedy mechanism like DQN did.

Same architecture idea works for both projects in this folder:
    - the tiny 1D world: input is a one-hot vector (n_states,)
    - CartPole:          input is the raw 4-dim continuous state

The network doesn't care which -- it just needs `state_size` inputs.
"""

import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        """state: (batch, state_size) -> action probabilities (batch, action_size)."""
        logits = self.network(state)
        return torch.softmax(logits, dim=-1)


if __name__ == "__main__":
    # Smoke test: does a forward pass give a valid probability distribution?
    policy = PolicyNetwork(state_size=5, action_size=2)
    one_hot_state = torch.zeros(1, 5)
    one_hot_state[0, 2] = 1.0  # state = 2

    probs = policy(one_hot_state)
    print("Action probabilities:", probs)
    print("Sums to 1:", torch.allclose(probs.sum(), torch.tensor(1.0)))

    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    print("Sampled action:", action.item(), "log_prob:", dist.log_prob(action).item())