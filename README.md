# Chef's Hat Gym — Opponent Modelling with DQN and PPO
## Module: 7043SCN — Generative AI and Reinforcement Learning
## Institution: Coventry University
## Author: Mohamed Bseikri

### Overview
This project investigates opponent modelling in the Chef's Hat card game by training reinforcement learning agents against different opponent types and measuring the impact on learned policy quality.

Two RL algorithms are compared:

- DQN (Deep Q-Network) — value-based, off-policy
- PPO (Proximal Policy Optimisation) — policy gradient, on-policy

Each algorithm is trained under three opponent conditions (6 experiments in total), all for 44,000 games to ensure a fair comparison.

### Prerequisites
- python=3.10.19
- torch==2.10.0
- gymnasium==1.2.3
- chefshatgym==3.0.0.1

### Agent Architecture

| Parameter | DQN | PPO |
| :--- | :--- | :--- |
| Learning rate | 1e-3 → 1e-5 | 3e-4 → 1e-5 |
| Gamma | 0.99 | 0.99 |
| Batch size | 128 | 64 |
| Hidden size | 256 | 256 |
| Update frequency | Every 10 games | Every 10 games |
| Games trained | 44,000 | 44,000 |
| Matches per game | 5 | 5 |

### Experimental Setup
