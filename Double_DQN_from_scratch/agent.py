"""
Double DQN. Same everything as Project 3's DQNAgent -- same network,
same replay buffer, same epsilon-greedy, same Huber loss -- except for
ONE line: how the Bellman target is computed.

### The problem, recap from Project 3

Vanilla DQN's target:

    y = r + gamma * max_a' Q_target(s', a')

`max_a'` does two jobs at once using the SAME network (the target net):
    1. SELECT which next action looks best
    2. EVALUATE how good that action actually is

If the target network's Q-value estimates have any noise (they always
do), `max` is a biased estimator: it systematically favors actions whose
value happens to be overestimated by noise, because those are the ones
that "win" the max. Errors don't cancel out -- they compound, because
next episode's target depends on this episode's (slightly too high)
Q-values, over and over. That's the upward-creeping loss and the reward
collapse you saw in Project 3's `results/loss.png`.

### The fix: decouple selection from evaluation

Double DQN target:

    a*  = argmax_a' Q_online(s', a')      <- ONLINE network selects the action
    y   = r + gamma * Q_target(s', a*)    <- TARGET network evaluates it

Now, for the target to be too high, the online network would have to
select an action AND the target network would have to independently
overestimate that same specific action -- much less likely than a single
network doing both jobs and reinforcing its own bias.

Compare this to Project 3's `learn()` -- everything is identical except
the four lines computing `next_q_values` inside `torch.no_grad()`.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork


class DoubleDQNAgent:
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
        target_update_freq=500,
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

        self.online_net = QNetwork(n_states, n_actions, hidden_size).to(self.device)
        self.target_net = QNetwork(n_states, n_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    def act(self, state):
        """Identical to Project 3: epsilon-greedy using the online network."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def learn(self, batch):
        """
        One training step. Identical to Project 3's DQNAgent.learn() except
        for how `next_q_values` (the target) is computed -- see the block
        marked "DOUBLE DQN" below and compare it to Project 3's single line:

            next_q_values = self.target_net(next_states).max(dim=1).values
        """
        states, actions, rewards, next_states, dones = batch

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # --- DOUBLE DQN: selection and evaluation use different networks ---
            # 1. SELECT the best next action using the ONLINE network.
            best_next_actions = self.online_net(next_states).argmax(dim=1)
            # 2. EVALUATE that specific action using the TARGET network.
            next_q_values = self.target_net(next_states).gather(
                1, best_next_actions.unsqueeze(1)
            ).squeeze(1)
            # (Project 3's vanilla DQN instead did, in one step:
            #    next_q_values = self.target_net(next_states).max(dim=1).values
            #  which lets the target network both select AND evaluate.)

            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, targets)

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
            {"online_state_dict": self.online_net.state_dict(), "epsilon": self.epsilon},
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["online_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)


if __name__ == "__main__":
    # Smoke test: confirm learn() runs and produces a finite, sane loss.
    agent = DoubleDQNAgent(n_states=4, n_actions=2, target_update_freq=5)

    fake_states = np.random.randn(32, 4).astype(np.float32)
    fake_actions = np.random.randint(0, 2, size=32).astype(np.int64)
    fake_rewards = np.ones(32, dtype=np.float32)
    fake_next_states = np.random.randn(32, 4).astype(np.float32)
    fake_dones = np.zeros(32, dtype=np.float32)

    for step in range(10):
        loss = agent.learn((fake_states, fake_actions, fake_rewards, fake_next_states, fake_dones))
        print(f"step {step}: loss = {loss:.4f}")

    print("act() on a single random state:", agent.act(np.random.randn(4).astype(np.float32)))