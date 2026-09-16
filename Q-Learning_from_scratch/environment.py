"""
A tiny GridWorld environment, built completely from scratch (no Gymnasium).
This is the "black box" you're supposed to understand fully.

Grid layout (3 rows x 4 cols):

    +---+---+---+---+
    | S |   |   |   |
    +---+---+---+---+
    |   | X |   |   |
    +---+---+---+---+
    |   |   |   | G |
    +---+---+---+---+

    S = Start   (row 0, col 0)
    G = Goal    (row 2, col 3)
    X = Obstacle(row 1, col 1)

Actions:
    0 = UP
    1 = RIGHT
    2 = DOWN
    3 = LEFT

Rewards:
    +10  reach goal
    -10  hit obstacle (episode ends)
    -1   normal move (encourages shorter paths)
"""

import numpy as np


class GridWorld:
    def __init__(self, n_rows=3, n_cols=4, start=(0, 0), goal=(2, 3), obstacles=None):
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles is not None else [(1, 1)]

        # Action -> (row_delta, col_delta)
        self.action_effects = {
            0: (-1, 0),  # UP
            1: (0, 1),   # RIGHT
            2: (1, 0),   # DOWN
            3: (0, -1),  # LEFT
        }
        self.action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

        self.n_states = n_rows * n_cols
        self.n_actions = len(self.action_effects)

        self.agent_pos = None

    # helpers to convert between (row, col) and a single integer state 
    def _to_state(self, pos):
        row, col = pos
        return row * self.n_cols + col

    def _to_pos(self, state):
        row = state // self.n_cols
        col = state % self.n_cols
        return (row, col)

    def reset(self):
        """Put the agent back at the start. Returns the initial state (int)."""
        self.agent_pos = self.start
        return self._to_state(self.agent_pos)

    def step(self, action):
        """
        Apply an action to the environment.

        Returns:
            next_state (int)
            reward (float)
            done (bool)
        """
        row, col = self.agent_pos
        d_row, d_col = self.action_effects[action]

        new_row = row + d_row
        new_col = col + d_col

        # If the move would leave the grid, stay in place (still costs -1)
        if not (0 <= new_row < self.n_rows) or not (0 <= new_col < self.n_cols):
            new_row, new_col = row, col

        new_pos = (new_row, new_col)

        if new_pos in self.obstacles:
            reward = -10
            done = True
            self.agent_pos = new_pos
        elif new_pos == self.goal:
            reward = 10
            done = True
            self.agent_pos = new_pos
        else:
            reward = -1
            done = False
            self.agent_pos = new_pos

        return self._to_state(self.agent_pos), reward, done

    def render(self):
        """Print the grid with the agent's current position marked as 'A'."""
        grid = [["." for _ in range(self.n_cols)] for _ in range(self.n_rows)]

        for (r, c) in self.obstacles:
            grid[r][c] = "X"

        gr, gc = self.goal
        grid[gr][gc] = "G"

        sr, sc = self.start
        if grid[sr][sc] == ".":
            grid[sr][sc] = "S"

        ar, ac = self.agent_pos
        grid[ar][ac] = "A"

        print("+" + "---+" * self.n_cols)
        for row in grid:
            print("| " + " | ".join(row) + " |")
            print("+" + "---+" * self.n_cols)


if __name__ == "__main__":
    # Quick manual smoke test: take a few random actions and render each step.
    env = GridWorld()
    state = env.reset()
    print("Initial state:", state)
    env.render()

    rng = np.random.default_rng(0)
    for _ in range(5):
        action = rng.integers(0, env.n_actions)
        next_state, reward, done = env.step(action)
        print(f"action={env.action_names[action]} -> state={next_state}, reward={reward}, done={done}")
        env.render()
        if done:
            break