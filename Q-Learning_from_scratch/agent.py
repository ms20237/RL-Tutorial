"""
A Q-learning agent built from scratch with a plain NumPy Q-table.
No neural network here yet -- that comes later (DQN project).

Core update rule (off-policy TD control):

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

"Off-policy" because the update uses max_a' Q(s',a') -- the BEST possible
next action -- regardless of which action the agent will actually take next.
"""
import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_states,
        n_actions,
        alpha=0.1,                      # learning rate
        gamma=0.99,                     # discount factor
        epsilon=1.0,                    # initial exploration rate
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

        # The Q-table: one row per state, one column per action.
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best known action (ties broken randomly for stability)
            best_actions = np.flatnonzero(self.Q[state] == self.Q[state].max())
            return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """One step of the Q-learning (Bellman) update."""
        current_q = self.Q[state, action]

        if done:
            target = reward  # no future reward once the episode has ended
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])

        td_error = target - current_q
        self.Q[state, action] = current_q + self.alpha * td_error

    def decay_epsilon(self):
        """Call once per episode to gradually shift from exploration to exploitation."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)