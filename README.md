[README.md](https://github.com/user-attachments/files/33675132/README.md)
# Artificial Intelligence Coursework

**Search algorithms, game-playing agents, Markov decision processes, and decision trees, implemented in Python.**

## Results at a Glance

| Project | Result |
|---|---|
| Path search on grid maps | A* found paths up to **32% cheaper** than BFS |
| ID3 decision tree (from scratch) | **94.69%** test accuracy |
| Game agents | Minimax, Alpha-Beta, and Expectimax running under a **1-second** time limit per move |

## Part 1 – Search Algorithms

A delivery driver needs the cheapest route on a grid map with obstacles, different terrain costs, and teleport passways.

- Implemented **BFS (graph search)**, **Greedy Best-First Search**, **A\***, and **A\*-epsilon**.
- Designed a heuristic using Manhattan distance capped by the passway cost, and proved it is **consistent**.
- Benchmarked all algorithms on 12x12, 15x15, and 20x20 maps.

| Algorithm | Path cost (20x20) | Nodes expanded (20x20) |
|---|---|---|
| BFS | 256 | 390 |
| Greedy Best-First | 268 | 46 |
| A* | **175** | 309 |
| A*-epsilon (ε = 0.5) | 175 | 455 |
| A*-epsilon (ε = 4) | 206 | 42 |

**Takeaway:** A* always finds the cheapest path. Greedy expands far fewer nodes but returns expensive paths. A*-epsilon trades a little path cost for much less search.

## Part 2 – Game-Playing Agents

Two robots compete on a warehouse grid to deliver packages and earn points while managing their battery.

- **Improved Greedy** agent with a custom heuristic based on battery, score, and distances to packages and charging stations.
- **Minimax** with iterative deepening, so it always has a move ready when time runs out.
- **Alpha-Beta pruning** to search deeper in the same time.
- **Expectimax** using known information about the opponent: it picks "move west" and "pick up" 3 times more often than other moves.

```bash
python main.py alphabeta greedy -t 1 -s 1234 -c 200 --screen_print
```

## Part 3 – Markov Decision Processes and Learning

- **Value Iteration**, **Policy Evaluation**, and **Policy Iteration** for grid worlds with random transitions.
- **First-Visit Monte Carlo** estimation of state values from recorded simulation episodes, compared across 10, 100, and 1,000 episodes.
- **ID3 decision tree** built from scratch, with entropy-based splits and dynamic thresholds for continuous features.
- A vectorized **pairwise Euclidean distance** function in NumPy, without explicit loops.

## Tech Stack

Python · NumPy · pandas · Jupyter

## Repository Structure

```
hw1-search/        # BFS, Greedy, A*, A*-epsilon
hw2-game-agents/   # Greedy, Minimax, Alpha-Beta, Expectimax
hw3-mdp-learning/  # Value/Policy Iteration, Monte Carlo, ID3
```

## About

Coursework for Introduction to Artificial Intelligence (236501) at the Technion, completed with a partner. The game environment, simulator, and environment setup code were provided by the course staff.

> If you're currently taking this course, please follow your course's academic integrity policy.
