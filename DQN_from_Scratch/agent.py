"""
This is where every DQN component comes together:

    - online network     (QNetwork)      -- what we actually train
    - target network     (QNetwork)      -- a lagging copy, for stable targets
    - epsilon-greedy                     -- exploration
    - Bellman target                     -- y = r + gamma * max_a' Q_target(s', a')
    - Huber loss + backprop              -- how the online network learns

Compare `act()` here to Project 1/2's `choose_action()`: same epsilon-greedy
idea, but "look up the best action in the Q-table" becomes "run the state
through the network and take argmax over the output."

Compare `learn()` here to Project 1's `update()`: same Bellman-equation
idea, but instead of nudging one table entry by `alpha * td_error`, we
compute a loss over a whole batch and let the optimizer do the nudging via
backpropagation.

### Why two networks?

If we used the SAME network to compute both the prediction Q(s,a) and the
target r + gamma * max Q(s',a'), then every gradient step would move the
target too -- the network would be chasing a target that runs away from it
every time it takes a step towards it. That's the "moving target" problem.

The target network is a snapshot of the online network's weights, frozen
for a while (`target_update_freq` steps), so the target stays still long
enough for the online network to actually converge towards it before it
gets updated again.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork


class DQNAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        hidden_size=64,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        target_update_freq=500,   # in gradient steps, not episodes
        device=None,
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_update_freq = target_update_freq
        self._train_steps = 0

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network: updated every training step.
        self.online_net = QNetwork(n_states, n_actions, hidden_size).to(self.device)
        # Target network: same architecture, updated only periodically.
        self.target_net = QNetwork(n_states, n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # never trained directly with backprop

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    def act(self, state):
        """Epsilon-greedy action selection using the ONLINE network."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def learn(self, batch):
        """
        One training step on a batch sampled from the replay buffer.
        batch = (states, actions, rewards, next_states, dones), all numpy arrays.
        Returns the scalar loss (float) for logging.
        """
        states, actions, rewards, next_states, dones = batch

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # current Q-values: Q_theta(s, a) for the actions actually taken 
        # online_net(states) has shape (batch, n_actions); gather() picks out
        # the column corresponding to each row's chosen action.
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Bellman target: y = r + gamma * max_a' Q_target(s', a') 
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)
            # (1 - dones) zeroes out the bootstrap term for terminal transitions,
            # exactly like the `if done: target = reward` branch in Project 1/2.

        loss = nn.functional.smooth_l1_loss(q_values, targets)  # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "epsilon": self.epsilon,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["online_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)


if __name__ == "__main__":
    # Smoke test: one fake batch through learn(), confirm loss is a finite number.
    agent = DQNAgent(n_states=4, n_actions=2, target_update_freq=5)

    fake_states = np.random.randn(32, 4).astype(np.float32)
    fake_actions = np.random.randint(0, 2, size=32).astype(np.int64)
    fake_rewards = np.ones(32, dtype=np.float32)
    fake_next_states = np.random.randn(32, 4).astype(np.float32)
    fake_dones = np.zeros(32, dtype=np.float32)

    for step in range(10):
        loss = agent.learn((fake_states, fake_actions, fake_rewards, fake_next_states, fake_dones))
        print(f"step {step}: loss = {loss:.4f}")

    print("act() on a single random state:", agent.act(np.random.randn(4).astype(np.float32)))