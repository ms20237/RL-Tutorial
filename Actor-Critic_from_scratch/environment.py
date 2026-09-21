"""
Two intentionally tiny environments, built from scratch (no Gymnasium),
exactly as the doc recommends: understand Policy Gradient on something
so simple there's nowhere for a bug to hide, before touching CartPole.

### SimpleEnv -- the base line-world

    [0] [1] [2] [3] [4]
     ^               ^
    start           goal

Actions:
    0 = LEFT
    1 = RIGHT

Reward:
    +10  reach goal (state 4)
    -1   every other step

Optimal behavior: RIGHT, RIGHT, RIGHT, RIGHT.

### ObstacleEnv -- harder variant

    S . . X . . G
    0 1 2 3 4 5 6

Same actions. Stepping onto the obstacle cell ends the episode with a
penalty, so the agent has to learn to go around it -- RIGHT, RIGHT,
then some detour before continuing to the goal, rather than blindly
always moving RIGHT.
"""


class SimpleEnv:
    def __init__(self, n_states=5):
        self.n_states = n_states
        self.n_actions = 2
        self.goal = n_states - 1

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if action == 0:  # LEFT
            self.state -= 1
        else:  # RIGHT
            self.state += 1

        self.state = max(0, min(self.state, self.n_states - 1))

        if self.state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        return self.state, reward, done

    def render(self):
        cells = ["."] * self.n_states
        cells[self.goal] = "G"
        cells[self.state] = "A" if self.state != self.goal else "A/G"
        print("[" + "] [".join(cells) + "]")


class ObstacleEnv:
    """
    "slightly harder" variant -- deliberately 2D, not 1D.

    On a strict 1D line, an obstacle sitting between start and goal can't
    actually be gone AROUND -- there's only one path, forwards or
    backwards, so "detour" isn't a real option (backing up just delays
    hitting it). A genuine detour needs a second dimension, so this
    reuses Project 1's GridWorld shape -- same 3x4 grid, same obstacle
    and goal placement -- but now controlled by a POLICY network instead
    of a Q-table:

        +---+---+---+---+
        | S |   |   |   |
        +---+---+---+---+
        |   | X |   |   |
        +---+---+---+---+
        |   |   |   | G |
        +---+---+---+---+

    Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT (matches Project 1 exactly).
    State is encoded as a single int (row * n_cols + col) so it can go
    through the same one_hot_encoder() used for SimpleEnv -- just with
    n_states = 12 instead of 5.
    """

    def __init__(self, n_rows=3, n_cols=4, start=(0, 0), goal=(2, 3), obstacle=(1, 1)):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal_pos = goal
        self.obstacle_pos = obstacle

        self.action_effects = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

        self.n_states = n_rows * n_cols
        self.n_actions = 4
        self.goal = self._to_state(goal)       # int, so evaluate.py can compare directly
        self.obstacle = self._to_state(obstacle)

    def _to_state(self, pos):
        row, col = pos
        return row * self.n_cols + col

    def _to_pos(self, state):
        return (state // self.n_cols, state % self.n_cols)

    def reset(self):
        self.agent_pos = self.start
        return self._to_state(self.agent_pos)

    def step(self, action):
        row, col = self.agent_pos
        d_row, d_col = self.action_effects[action]
        new_row, new_col = row + d_row, col + d_col

        if not (0 <= new_row < self.n_rows) or not (0 <= new_col < self.n_cols):
            new_row, new_col = row, col  # bump into the wall, stay put

        new_pos = (new_row, new_col)

        if new_pos == self.obstacle_pos:
            reward, done = -10, True
        elif new_pos == self.goal_pos:
            reward, done = 10, True
        else:
            reward, done = -1, False

        self.agent_pos = new_pos
        return self._to_state(self.agent_pos), reward, done

    def render(self):
        grid = [["." for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        gr, gc = self.goal_pos
        grid[gr][gc] = "G"
        orr, oc = self.obstacle_pos
        grid[orr][oc] = "X"
        ar, ac = self.agent_pos
        grid[ar][ac] = "A"

        print("+" + "---+" * self.n_cols)
        for row in grid:
            print("| " + " | ".join(row) + " |")
            print("+" + "---+" * self.n_cols)


if __name__ == "__main__":
    print("SimpleEnv smoke test:")
    env = SimpleEnv()
    state = env.reset()
    env.render()
    for action in [1, 1, 1, 1]:  # always RIGHT
        state, reward, done = env.step(action)
        env.render()
        print(f"  action=RIGHT -> state={state}, reward={reward}, done={done}")
        if done:
            break

    print("\nObstacleEnv smoke test (deliberately walking into the obstacle):")
    env2 = ObstacleEnv()
    env2.reset()
    env2.render()
    for action in [2, 1]:  # DOWN then RIGHT -> (0,0) -> (1,0) -> (1,1) = obstacle
        state, reward, done = env2.step(action)
        env2.render()
        print(f"  action={env2.action_names[action]} -> state={state}, reward={reward}, done={done}")
        if done:
            break

    print("\nObstacleEnv smoke test (a working detour: RIGHT, RIGHT, DOWN, DOWN, RIGHT):")
    env3 = ObstacleEnv()
    env3.reset()
    env3.render()
    for action in [1, 1, 2, 2, 1]:
        state, reward, done = env3.step(action)
        env3.render()
        print(f"  action={env3.action_names[action]} -> state={state}, reward={reward}, done={done}")
        if done:
            break