"""
Two small networks, matching the roadmap's diagram exactly:

                     State
                       |
              +--------+--------+
              |                 |
              v                 v
            Actor             Critic
              |                 |
              v                 v
            Action             Value

    Actor  (ActorNetwork):  pi_theta(a|s)  -- same shape as Project 5's
                             PolicyNetwork: state -> softmax over actions.

    Critic (CriticNetwork):  V_phi(s)       -- NEW in this project: state ->
                             a single scalar, estimating "how good is it
                             to be in this state, under the current policy."

Why do we need the critic at all? Project 5's REINFORCE used the raw
episode return G_t as the learning signal, which has to wait for the
whole episode to finish and carries a lot of noise (see Project 5's
README on why that caused the policy to collapse on the obstacle task).
The critic gives us a running ESTIMATE of value that can be updated
every single step, without waiting for the episode to end -- that's
what makes an online, step-by-step update possible.
"""
import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """Identical architecture to Project 5's PolicyNetwork: state -> pi_theta(a|s)."""

    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """state -> V(s), a single scalar. No softmax -- a value isn't a probability."""

    def __init__(self, state_size, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)  # (batch, 1) -> (batch,)


if __name__ == "__main__":
    # Smoke test: both networks on the same fake batch of one-hot states.
    actor = ActorNetwork(state_size=5, action_size=2)
    critic = CriticNetwork(state_size=5)

    state = torch.zeros(1, 5)
    state[0, 2] = 1.0

    probs = actor(state)
    value = critic(state)
    print("Actor probabilities:", probs, "sums to 1:", torch.allclose(probs.sum(), torch.tensor(1.0)))
    print("Critic value V(s):", value.item())