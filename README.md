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


### Experimental Setup
