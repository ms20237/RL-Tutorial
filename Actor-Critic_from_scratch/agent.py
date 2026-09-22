"""
One-step (TD) Actor-Critic. The key structural difference from Project
5's REINFORCE: we no longer wait for a whole episode to finish before
learning. Every single step gives us enough information to update both
networks immediately.

### The critic's target and loss

The critic learns to predict V(s) -- the expected return from state s.
Its target, for one step, is the TD target:

    y = r + gamma * V(s')      (or just r if s' is terminal)

which is a ONE-STEP estimate of the return, using the critic's own
(current) estimate of V(s') to stand in for everything that would happen
after this step -- exactly the same bootstrapping idea Project 3's DQN
used for Q-values, just applied to state values instead of state-action
values.

    critic_loss = (y - V(s))^2          <- ordinary regression / MSE

### The actor's loss and the advantage

Project 5's REINFORCE used the raw return G_t as the "how good was this
action" signal:

    actor_loss = -log pi(a|s) * G_t

The problem: G_t's absolute scale carries no information about whether
an action was better or worse than what you'd expect from that state
anyway -- a return of +7 might be great from a bad state or mediocre
from a great one. Actor-Critic replaces G_t with the ADVANTAGE:

    A(s,a) = y - V(s)     =     r + gamma * V(s') - V(s)

This is just the critic's TD error. It answers a sharper question: "was
this action better or worse than the critic's own expectation for this
state?" That's a much lower-variance, more informative signal than the
raw return, which is exactly what fixes the instability you saw with
REINFORCE.

    actor_loss = -log pi(a|s) * A(s,a)

Notice `A(s,a)` is `.detach()`-ed before it's used in the actor loss --
we want the advantage's VALUE to scale the actor's gradient, but we
don't want gradients from the actor loss flowing backward into the
critic. The critic is trained by its own loss, independently.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from networks import ActorNetwork, CriticNetwork


class ActorCriticAgent:
    def __init__(
        self,
        state_size,
        action_size,
        state_encoder,
        hidden_size=32,
        actor_lr=1e-3,
        critic_lr=1e-2,   # critic usually wants a faster learning rate than the actor
        gamma=0.99,
        device=None,
    ):
        self.state_encoder = state_encoder
        self.gamma = gamma

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(state_size, action_size, hidden_size).to(self.device)
        self.critic = CriticNetwork(state_size, hidden_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        """Same idea as Project 5: sample from pi_theta, don't argmax."""
        state_t = self.state_encoder(state).to(self.device).unsqueeze(0)
        probs = self.actor(state_t)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return int(action.item()), log_prob.squeeze(0)

    def learn(self, state, log_prob, reward, next_state, done):
        """
        One ONLINE update, called after every single environment step (not
        once per episode, unlike Project 5). Returns (actor_loss, critic_loss)
        as floats, for logging.
        """
        state_t = self.state_encoder(state).to(self.device).unsqueeze(0)
        next_state_t = self.state_encoder(next_state).to(self.device).unsqueeze(0)

        value = self.critic(state_t).squeeze(0)               # V(s), WITH grad
        with torch.no_grad():
            next_value = self.critic(next_state_t).squeeze(0)  # V(s'), no grad -- it's a target
            td_target = reward + self.gamma * next_value * (0.0 if done else 1.0)
            advantage = td_target - value  # detached: used only to scale the actor's gradient

        # Critic update: ordinary regression towards the TD target 
        critic_loss = nn.functional.mse_loss(value, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update: policy gradient, scaled by the (detached) advantage 
        actor_loss = -log_prob * advantage
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])


# Reuse the same ready-made state encoders as Project 5 

def one_hot_encoder(n_states):
    def encode(state):
        vec = torch.zeros(n_states, dtype=torch.float32)
        vec[state] = 1.0
        return vec
    return encode


def identity_encoder(state):
    return torch.as_tensor(state, dtype=torch.float32)


if __name__ == "__main__":
    # Smoke test: one fake episode, step-by-step online updates.
    encoder = one_hot_encoder(n_states=5)
    agent = ActorCriticAgent(state_size=5, action_size=2, state_encoder=encoder)

    fake_transitions = [
        (0, 1, -1, False),
        (1, 2, -1, False),
        (2, 3, -1, False),
        (3, 4, 10, True),
    ]

    for state, next_state, reward, done in fake_transitions:
        action, log_prob = agent.select_action(state)
        actor_loss, critic_loss = agent.learn(state, log_prob, reward, next_state, done)
        print(f"state={state} -> action={action}, actor_loss={actor_loss:.3f}, critic_loss={critic_loss:.3f}")