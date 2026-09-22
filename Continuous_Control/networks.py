"""
Two actors, one shared critic design, to make the discrete-vs-continuous
comparison this project is built around as apples-to-apples as possible.

### DiscreteActor -- identical idea to Project 6's ActorNetwork

    state -> logits -> softmax -> pi(a|s) over a FIXED MENU of actions
                                   {u=-1, u=0, u=+1}

### GaussianActor -- the new piece: a policy over a CONTINUOUS action

Instead of a probability for each of a few fixed choices, the network
outputs the two numbers that define a continuous probability
distribution over u in [-1, 1]:

    mean  = tanh(raw_mean)     -- squashed into [-1, 1], since u is bounded
    std   = softplus(raw_std)  -- must be positive; softplus guarantees that

    pi_theta(u|s) = Normal(mean, std)

We then SAMPLE u from that Normal distribution (clipped to [-1,1] before
it's sent to the environment), exactly the same "sample, don't argmax"
idea as the discrete case -- just with a continuous distribution instead
of a categorical one. log_prob(u) is computed from the (unclipped)
Normal distribution and used in the policy-gradient loss exactly the
way log_prob(action) was used for the discrete case.

### Critic -- exactly the same for both

V(s) doesn't care whether the action space is discrete or continuous;
it only ever looks at the STATE. One shared class works for both agents.
"""
import torch
import torch.nn as nn


class DiscreteActor(nn.Module):
    def __init__(self, state_size, n_actions, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, state):
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)


class GaussianActor(nn.Module):
    """
    Outputs only the MEAN of the action distribution; the standard
    deviation is handled separately by the agent (see the note in
    agent.py on why). This is a deliberate simplification: letting a
    network learn its own std, with no other stabilization, is prone to
    a specific failure mode covered in this project's README -- log_std
    saturating at its ceiling after one unlucky gradient step and never
    coming back down. A fixed, manually-decayed std sidesteps that
    entirely and is standard practice for a first from-scratch
    continuous-control implementation.
    """

    def __init__(self, state_size, action_size=1, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state):
        """Returns mean, shape (batch, action_size), squashed into [-1, 1]."""
        return torch.tanh(self.net(state))


class CriticNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)


if __name__ == "__main__":
    state = torch.tensor([[1.5, -0.3]])  # [x, v]

    discrete_actor = DiscreteActor(state_size=2, n_actions=3)
    probs = discrete_actor(state)
    print("Discrete actor probs:", probs, "sums to 1:", torch.allclose(probs.sum(), torch.tensor(1.0)))

    gaussian_actor = GaussianActor(state_size=2, action_size=1)
    mean = gaussian_actor(state)
    std = torch.tensor([[0.3]])  # in this design, std is supplied by the agent, not the network
    print("Gaussian actor mean:", mean.item(), "(std supplied externally, e.g. 0.3)")

    dist = torch.distributions.Normal(mean, std)
    u = dist.sample()
    print("Sampled u (before clipping):", u.item(), "log_prob:", dist.log_prob(u).item())

    critic = CriticNetwork(state_size=2)
    print("V(s):", critic(state).item())