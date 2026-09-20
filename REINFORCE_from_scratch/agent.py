"""
The REINFORCE agent. This file is deliberately generic over how a raw
environment state gets turned into a tensor (`state_encoder`), so the
EXACT SAME agent class works for:

    - the tiny 1D world (state_encoder = one-hot encoding)
    - CartPole            (state_encoder = identity, state is already 4 floats)

This mirrors the doc's point in section 17: "the rest of your algorithm
remains essentially the same" once you move from the toy world to
CartPole. Only the environment and the policy network's input size
change; the agent logic doesn't.

Five things this file implements, matching the doc's "five things you
really need to understand":

    pi_theta(a|s)                         -> select_action()
    G_t                                   -> compute_returns()
    log pi_theta(a_t|s_t)                 -> the log_prob returned by select_action()
    L = -G_t * log pi_theta(a_t|s_t)      -> update()
    theta <- theta - alpha * grad(L)      -> optimizer.step() inside update()
"""

import torch
import torch.optim as optim

from policy import PolicyNetwork


def compute_returns(rewards, gamma):
    """
    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    Computed backwards through the episode so each G_t can reuse the
    G_{t+1} that was just computed: G_t = r_t + gamma * G_{t+1}.
    """
    returns = []
    G = 0.0
    for reward in reversed(rewards):
        G = reward + gamma * G
        returns.insert(0, G)
    return returns


class REINFORCEAgent:
    def __init__(
        self,
        state_size,
        action_size,
        state_encoder,
        hidden_size=32,
        lr=1e-2,
        gamma=0.99,
        normalize_returns=False,
        device=None,
    ):
        """
        state_encoder: a function raw_state -> torch.Tensor of shape (state_size,)
                       e.g. one-hot encoding for the 1D world, or just
                       torch.as_tensor(state, dtype=torch.float32) for CartPole.
        normalize_returns: subtract the mean and divide by std of the
                       episode's returns before using them in the loss.
                       Not part of vanilla REINFORCE, but a very common
                       and cheap variance-reduction trick worth trying
                       once the raw version works (see README).
        """
        self.state_encoder = state_encoder
        self.gamma = gamma
        self.normalize_returns = normalize_returns

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(state_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        """
        Sample an action from pi_theta(.|s) -- NOT argmax. Returns both the
        action (to send to env.step()) and its log-probability (needed
        later for the policy-gradient loss).
        """
        state_t = self.state_encoder(state).to(self.device).unsqueeze(0)
        probs = self.policy(state_t)  # shape (1, action_size)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), log_prob.squeeze(0)

    def update(self, log_probs, rewards):
        """
        The REINFORCE update, run once per COMPLETE episode (not per step --
        we need the whole trajectory to compute returns).

            L(theta) = -sum_t G_t * log pi_theta(a_t | s_t)

        The minus sign turns PyTorch's gradient DESCENT into gradient
        ASCENT on expected return: increasing the log-probability of
        actions that led to high returns, decreasing it for actions that
        led to low (or negative) returns.
        """
        returns = compute_returns(rewards, self.gamma)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if self.normalize_returns and len(returns_t) > 1:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = 0.0
        for log_prob, G in zip(log_probs, returns_t):
            loss += -log_prob * G

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path, map_location=self.device))


# Ready-made state encoders --------------------------------------------------

def one_hot_encoder(n_states):
    """Returns a function: int state -> one-hot torch.Tensor of shape (n_states,)."""
    def encode(state):
        vec = torch.zeros(n_states, dtype=torch.float32)
        vec[state] = 1.0
        return vec
    return encode


def identity_encoder(state):
    """For environments whose state is already a flat vector of floats (e.g. CartPole)."""
    return torch.as_tensor(state, dtype=torch.float32)


if __name__ == "__main__":
    # Smoke test: run one fake episode through select_action() and update().
    encoder = one_hot_encoder(n_states=5)
    agent = REINFORCEAgent(state_size=5, action_size=2, state_encoder=encoder)

    log_probs, rewards = [], []
    fake_states = [0, 1, 2, 3]
    fake_rewards = [-1, -1, -1, 10]

    for state, reward in zip(fake_states, fake_rewards):
        action, log_prob = agent.select_action(state)
        print(f"state={state} -> action={action}, log_prob={log_prob.item():.3f}")
        log_probs.append(log_prob)
        rewards.append(reward)

    loss = agent.update(log_probs, rewards)
    print("Loss:", loss)
    print("Returns for rewards", fake_rewards, "=", compute_returns(fake_rewards, gamma=0.99))