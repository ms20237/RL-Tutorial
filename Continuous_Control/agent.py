"""
Two agents, built to make the discrete-vs-continuous comparison direct:
- DiscreteACAgent: same algorithm as Project 6's ActorCriticAgent, applied
  to DiscretizedPointMassEnv's 3-action menu.
- ContinuousACAgent: the new piece. Structurally almost identical --
  same critic, same TD target, same advantage, same "actor_loss =
  -log_prob * advantage" -- the ONLY difference is how log_prob is
  computed: from a Normal distribution instead of a Categorical one.

That similarity is the point: the core Actor-Critic algorithm doesn't
change at all when the action space goes from discrete to continuous.
Only the actor's output distribution changes.

### Why both learn() methods clip gradients (max_norm=1.0)

This wasn't in the original design -- it was added after hitting a real
failure while building this project. With a CONTINUOUS Gaussian policy,
log_prob(u) can swing much further than a discrete Categorical's
log_prob ever can (a categorical log_prob is bounded by how many
actions you have; a Gaussian log_prob is not bounded at all -- it grows
without limit as the policy's std shrinks and/or the sampled action
lands far from the mean). Combine one occasional big log_prob with the
environment's -50 divergence penalty creating a large advantage, and a
single unlucky step can produce a huge gradient that permanently
saturates the policy's log_std at its clamp ceiling -- after which the
policy stays maximally noisy forever and can never recover precise
control. See README.md for the full story; gradient clipping is the
standard, direct fix.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks import DiscreteActor, GaussianActor, CriticNetwork


def identity_encoder(state):
    return torch.as_tensor(state, dtype=torch.float32)


class DiscreteACAgent:
    def __init__(self, state_size, n_actions, hidden_size=64, actor_lr=1e-3, critic_lr=1e-2,
                 gamma=0.99, device=None):
        self.gamma = gamma
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = DiscreteActor(state_size, n_actions, hidden_size).to(self.device)
        self.critic = CriticNetwork(state_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        state_t = identity_encoder(state).to(self.device).unsqueeze(0)
        probs = self.actor(state_t)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze(0)

    def learn(self, state, log_prob, reward, next_state, done):
        state_t = identity_encoder(state).to(self.device).unsqueeze(0)
        next_state_t = identity_encoder(next_state).to(self.device).unsqueeze(0)

        value = self.critic(state_t).squeeze(0)
        with torch.no_grad():
            next_value = self.critic(next_state_t).squeeze(0)
            td_target = reward + self.gamma * next_value * (0.0 if done else 1.0)
            advantage = td_target - value

        critic_loss = nn.functional.mse_loss(value, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = -log_prob * advantage
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


class ContinuousACAgent:
    def __init__(self, state_size, action_size=1, hidden_size=64, actor_lr=1e-3, critic_lr=1e-2,
                 gamma=0.99, std_start=0.6, std_min=0.05, std_decay=0.995, device=None):
        """
        std_start / std_min / std_decay: the action-distribution's standard
        deviation is a fixed, manually-decayed number (not learned by the
        network -- see the note in networks.py's GaussianActor). This is
        exactly the same idea as DQN's epsilon decay (Project 3): start
        with a lot of exploration noise, and shrink it over training as
        the policy's mean becomes more trustworthy. Call decay_std() once
        per episode, same as DQN's decay_epsilon().
        """
        self.gamma = gamma
        self.std = std_start
        self.std_min = std_min
        self.std_decay = std_decay

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = GaussianActor(state_size, action_size, hidden_size).to(self.device)
        self.critic = CriticNetwork(state_size, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        """
        Sample a CONTINUOUS action u from a Normal(mean, self.std) distribution.
        Returns:
            u_clipped: float in [-1, 1], to send to env.step()
            log_prob:  log pi(u|s), computed from the UNCLIPPED sample --
                       clipping only happens for the environment, not for
                       the gradient computation.
        """
        state_t = identity_encoder(state).to(self.device).unsqueeze(0)
        mean = self.actor(state_t)
        std = torch.full_like(mean, self.std)
        dist = torch.distributions.Normal(mean, std)

        raw_action = dist.sample()               # shape (1, 1)
        log_prob = dist.log_prob(raw_action).sum(dim=-1)  # sum over action dims (here just 1)

        u = float(torch.clamp(raw_action, -1.0, 1.0).item())
        return u, log_prob.squeeze(0)

    def learn(self, state, log_prob, reward, next_state, done):
        """Identical structure to DiscreteACAgent.learn() -- only log_prob's origin differs."""
        state_t = identity_encoder(state).to(self.device).unsqueeze(0)
        next_state_t = identity_encoder(next_state).to(self.device).unsqueeze(0)

        value = self.critic(state_t).squeeze(0)
        with torch.no_grad():
            next_value = self.critic(next_state_t).squeeze(0)
            td_target = reward + self.gamma * next_value * (0.0 if done else 1.0)
            advantage = td_target - value

        critic_loss = nn.functional.mse_loss(value, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        actor_loss = -log_prob * advantage
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def decay_std(self):
        """Call once per episode -- same role as DQN's decay_epsilon()."""
        self.std = max(self.std_min, self.std * self.std_decay)

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


if __name__ == "__main__":
    print("DiscreteACAgent smoke test:")
    d_agent = DiscreteACAgent(state_size=2, n_actions=3)
    for state, next_state, reward, done in [((1.0, 0.0), (0.95, -0.1), -1.0, False), ((0.95, -0.1), (0.85, -0.2), -0.9, True)]:
        action, log_prob = d_agent.select_action(np.array(state, dtype=np.float32))
        al, cl = d_agent.learn(np.array(state, dtype=np.float32), log_prob, reward, np.array(next_state, dtype=np.float32), done)
        print(f"  state={state} -> action={action}, actor_loss={al:.3f}, critic_loss={cl:.3f}")

    print("\nContinuousACAgent smoke test:")
    c_agent = ContinuousACAgent(state_size=2, action_size=1)
    for state, next_state, reward, done in [((1.0, 0.0), (0.95, -0.1), -1.0, False), ((0.95, -0.1), (0.85, -0.2), -0.9, True)]:
        u, log_prob = c_agent.select_action(np.array(state, dtype=np.float32))
        al, cl = c_agent.learn(np.array(state, dtype=np.float32), log_prob, reward, np.array(next_state, dtype=np.float32), done)
        print(f"  state={state} -> u={u:.3f}, actor_loss={al:.3f}, critic_loss={cl:.3f}")