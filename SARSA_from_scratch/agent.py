"""
SARSA: on-policy TD control.

Core update rule:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]

Compare this to Q-learning's update:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

The ONLY difference is the target:

    Q-learning: gamma * max_a' Q(s', a')   <- the BEST possible next action
                                               (whether or not the agent will
                                               actually take it) -> off-policy

    SARSA:      gamma * Q(s', a')          <- the action the agent's CURRENT
                                               policy actually picked next
                                               -> on-policy

That's why SARSA needs to know the next action *before* it can update --
it literally needs a whole (s, a, r, s', a') tuple, which is where the name
comes from: State, Action, Reward, State', Action'.
"""
import numpy as np


class SarsaAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    ):
        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """Epsilon-greedy action selection (identical to Q-learning)."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            best_actions = np.flatnonzero(self.Q[state] == self.Q[state].max())
            return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update. Note the extra `next_action` argument compared to
        Q-learning's `update()` -- SARSA needs to know what action the
        policy will actually take next, not just the best possible one.
        """
        current_q = self.Q[state, action]

        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]

        td_error = target - current_q
        self.Q[state, action] = current_q + self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)