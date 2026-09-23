# RL-Tutorial

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Gymnasium](https://img.shields.io/badge/Gymnasium-CartPole--v1-green)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-blue)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![No SB3](https://img.shields.io/badge/No%20Stable--Baselines3-From%20Scratch-success)

## 📖 Overview

This repository is a **from-scratch reinforcement learning curriculum** — seven projects built up in order, from a 12-state Q-table to a continuous-control Actor-Critic that matches a hand-designed PD controller. Every algorithm is written by hand. No Stable-Baselines3, no RLlib, no Gymnasium environment for the early projects. The only dependencies are PyTorch (as an autodiff library, not as an RL framework), NumPy, and Matplotlib.

The arc is deliberate: **tabular methods first, function approximation second, policy gradients third.** Each project inherits the environment and the training loop of the one before it, changing as little as possible so the algorithmic difference is the only thing that varies. By Project 4 you can point to *one line* that separates two algorithms. By Project 7 you can point to *one distribution* that separates discrete from continuous control.

Along the way, several things genuinely break — and those breaks are documented honestly, because they're the more useful part. REINFORCE permanently collapses on a simple obstacle grid. Vanilla DQN's Q-values drift upward and its policy falls apart. Actor-Critic fixes the obstacle task but introduces a 450-episode stall on CartPole. A continuous Gaussian policy saturates its own `log_std` at the clamp ceiling after one unlucky gradient. Every one of those failures is written up, with the plot or log that shows it, in the project's own README.

## 📂 Repository Structure

```
RL-Tutorial/
├── Q-Learning_from_scratch/          # Project 1 — off-policy tabular TD control
├── SARSA_from_scratch/               # Project 2 — on-policy tabular TD control
├── DQN_from_Scratch/                 # Project 3 — deep Q-network on CartPole
├── Double_DQN_from_scratch/          # Project 4 — the one-line overestimation fix
├── REINFORCE_from_scratch/           # Project 5 — policy gradients, from scratch
├── Actor-Critic_from_scratch/        # Project 6 — per-step advantage, V(s) critic
├── Continuous_Control/               # Project 7 — Gaussian policy, PD baseline
├── .gitignore
├── LICENSE
└── README.md
```

Each project folder is self-contained: its own `environment.py`, its own `agent.py`, its own training and evaluation scripts, its own `results/` and `checkpoints/`. Some projects need the sibling folder alongside them (Project 4's `compare.py` reads Project 3's results; Project 6's `compare_obstacle.py` reads Project 5's) — the READMEs say so where it applies.

## 🗺️ The Roadmap

### Project 1 — [Q-Learning from Scratch](Q-Learning_from_scratch)

The first algorithm, on a 3×4 GridWorld with 12 enumerable states and a 12×4 Q-table. Off-policy TD control: the update uses `max_a' Q(s',a')` for the next state, whether or not the agent will actually go there.

**Environment:** hand-rolled GridWorld with one obstacle.
**Learns:** optimal path `S → G` in 5 moves, total reward +5.
**Interesting failure:** none — this one works, which is why it's the first.

### Project 2 — [SARSA from Scratch](SARSA_from_scratch)

Same environment, same hyperparameters, same ε-greedy. **One change to the update:** the target uses `Q(s',a')` for the action the policy *actually picked*, not the max. That single change is the entire difference between off-policy and on-policy learning.

**Interesting finding:** on this deterministic grid the greedy path is identical, but the *policy* diverges in two cells — SARSA hedges away from the obstacle more strongly in cells adjacent to it, because its update accounts for the chance that exploration will send it there.

### Project 3 — [DQN from Scratch](DQN_from_Scratch)

The first project where a Q-table stops making sense. CartPole's state is four continuous numbers; there's nothing to look up. A `4 → 64 → 64 → 2` MLP replaces the table, a replay buffer decorrelates experience, a target network stabilizes the target, and Huber loss keeps outliers in check.

**Interesting failure, documented in full:** reward climbs from ~15 to a peak of ~164 around episode 210, then **collapses** back to ~50–95. The loss curve climbs steadily through the same period. This is the textbook DQN overestimation problem — `max` in the Bellman target is a biased estimator, so Q-values drift upward.

### Project 4 — [Double DQN from Scratch](Double_DQN_from_scratch)

The fix for Project 3. **One block of code changes** in `learn()`: the online network *selects* the best next action, the target network *evaluates* it. Selection and evaluation are no longer done by the same network.

**Result:** peak reward 284 → 332, final 50-episode average 71.7 → 96.1. Greedy evaluation improves from ~95 to ~109 average steps.
**Honest caveat:** the loss curves look *similar* — Double DQN didn't eliminate instability in this run, it got more useful learning out of the same amount of instability.

### Project 5 — [REINFORCE from Scratch](REINFORCE_from_scratch)

The first project with no Q-values at all. The policy itself is the network, exploration comes from sampling the policy's own distribution, and the update ascends the gradient of expected return using the raw Monte Carlo return `G_t`.

**Interesting failure #1:** `ObstacleEnv` — REINFORCE goes flat at −11 reward by episode ~300 and stays there for 1700 more episodes, 0% success, permanently.
**Interesting failure #2:** `CartPole` — repeatedly hits a perfect 500, repeatedly collapses back under 100, oscillating between solved and random for the whole run. Same underlying problem, two different manifestations.

### Project 6 — [Actor-Critic from Scratch](Actor-Critic_from_scratch)

Adds a critic `V(s)` and replaces the raw return with the advantage `A = r + γV(s') − V(s)`. Both networks update **every step**, not once per episode.

**The payoff:** on the same obstacle grid that permanently broke REINFORCE, Actor-Critic hits **100% success** and stays there. `comparison_obstacle.png` is the cleanest before/after in the whole course.
**Honest caveat again:** on CartPole, Actor-Critic climbs smoothly to ~370 average — then collapses to a flat ~9–10 for ~450 episodes before recovering. A real limitation of single-environment, single-sample online updates, and exactly the motivation for the parallel-environment and clipped-update fixes that come next.

### Project 7 — [Continuous Control](Continuous_Control)

The bridge from control theory into DRL. A 1D point mass (double integrator), an LQR-shaped reward, and three controllers compared from the same starting position: a hand-designed PD baseline, a discrete-action Actor-Critic, and a genuinely continuous Gaussian-policy Actor-Critic.

**The headline:** the learned continuous controller matches the PD baseline within 1%, having learned entirely from trial and error.
**The visible cost of discretization:** `comparison_trajectories.png`'s control-signal panel shows Discrete AC *chattering* between `u = −1` and `u = +1` for the rest of the trajectory, because with only three actions it can't express "hold near zero."
**Interesting failure:** the first continuous design let the network learn its own `log_std`. One unlucky gradient saturated it at the clamp ceiling permanently, every episode diverged, and the policy never recovered. The fix was a design change — fixed, manually-decayed `std`, plus gradient clipping — not a tuning tweak.

## 🧭 How the Projects Fit Together

Read in order, the projects form a chain of *single* changes, each isolating one algorithmic idea:

| Project | What changes from the previous | What stays the same |
| :--- | :--- | :--- |
| 2 (SARSA) | `max_a' Q(s',a')` → `Q(s',a')` | Same environment, same loop, same hyperparameters |
| 3 (DQN) | Q-table → neural network; + replay buffer, + target net | Same Bellman idea |
| 4 (Double DQN) | Target split across two networks | Same network, same buffer, same loop |
| 5 (REINFORCE) | Q-values → direct policy; + sampling for exploration | Same optimizer, same MLP shape |
| 6 (Actor-Critic) | Return `G_t` → advantage `A_t`; per-step updates | Same policy network, same update shape |
| 7 (Continuous) | Categorical policy → Gaussian policy | Same critic, same advantage, same actor loss |

That's the whole point of building it this way. Each step is small enough to see exactly what it buys and exactly what it costs.

## 🧵 Common Threads

A few things cut across all seven projects:

*   **Every environment is written by hand** except CartPole (Projects 3–6). The point is that you understand the state space, the action space, and the reward function completely before you start confusing environment behavior with algorithm behavior.
*   **Every module ships with a smoke test** — an `if __name__ == "__main__":` block that runs the file standalone. That's the "build in stages" workflow: verify the network in isolation, then the buffer, then the agent, then wire them together.
*   **Every project logs per-episode metrics to CSV** and ships a plotting script. The plots are the artifact of the project, not the code — they're what you look at when you want to know whether it actually learned.
*   **Failures are documented, not hidden.** The READMEs explain what went wrong, why, and what the fix was — or why the fix is a *different algorithm*, as in Projects 5 → 6 and 6 → 8.

## 🛠️ Technologies Used

*   **Python 3.9+**
*   **NumPy** — arrays, tabular Q-tables, dynamics, rolling averages
*   **PyTorch** — networks, autograd, `torch.distributions`, optimizers
*   **Gymnasium** — `CartPole-v1` (classic-control extra), used from Project 3 onward
*   **Matplotlib** — every reward curve, policy plot, and comparison chart
*   **csv** (stdlib) — per-episode logging
*   **No Stable-Baselines3, no RLlib, no `torchrl`** — every algorithm is hand-written

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ms20237/RL-Tutorial.git
    cd RL-Tutorial
    ```

2.  **Install dependencies once** (all projects share the same short list):
    ```bash
    pip install torch numpy matplotlib "gymnasium[classic-control]"
    ```

3.  **Run any project from its own folder.** For example, the tabular ones:
    ```bash
    cd Q-Learning_from_scratch
    python train.py
    python visualize.py
    ```

    The deep ones, from Project 3 onward:
    ```bash
    cd DQN_from_Scratch
    python train.py
    python plot_results.py
    python evaluate.py
    ```

4.  **The comparison scripts** need the sibling project alongside them:
    ```bash
    cd SARSA_from_scratch && python compare.py       # needs Q-Learning_from_scratch
    cd Double_DQN_from_scratch && python compare.py  # needs DQN_from_Scratch
    cd Actor-Critic_from_scratch && python compare_obstacle.py   # needs REINFORCE_from_scratch
    ```

Each project's own README has the full run instructions for that folder.

## 🎓 What This Is Meant to Teach

Working through all seven, in order, is meant to leave you with a mental model of reinforcement learning that isn't just a list of algorithm names. Specifically:

*   You should be able to explain why Q-Learning and SARSA differ in the policy they converge to, and *see* the difference in the arrows they print for a small grid.
*   You should be able to explain what goes wrong with vanilla DQN, point at the loss curve that shows it, and say precisely which line of code Double DQN changes.
*   You should be able to explain why REINFORCE fails and why Actor-Critic fixes it, and know that "variance in the gradient estimate" is the whole story.
*   You should be able to explain why a continuous policy needs a different output distribution than a discrete one, and why a Gaussian's `log_prob` is the source of a specific class of instability that a Categorical's isn't.
*   You should know, from having watched them, that RL algorithms fail in ways that are *structural*, not incidental — and that the next algorithm on the roadmap is usually the response to a specific failure of the previous one.

## 🔮 What Comes Next

Project 7 is where this repository currently ends. The natural continuation, and what the failure modes in Projects 5–7 point at, is:

*   **PPO** — clipped surrogate objective, parallel environments. Directly targets both the CartPole collapse in Project 6 and the runaway-gradient failure in Project 7.
*   **SAC** — off-policy continuous control with entropy regularization and twin critics. The standard benchmark algorithm for the kind of problem Project 7 introduces.
*   **TD3** — deterministic policy gradients with twin critics and delayed updates; the other major off-policy continuous-control algorithm.
*   **GAE** — generalized advantage estimation, which sits between Project 5's full Monte Carlo return and Project 6's single-step TD error.

If you're reading this to learn, the right order is still the order in the repository: tabular first, then deep Q-learning, then policy gradients, then continuous control. Skipping ahead makes the later projects look like magic. Reading them in order makes them look like the obvious next thing to try.

## License

This project is licensed under the [MIT License](LICENSE).