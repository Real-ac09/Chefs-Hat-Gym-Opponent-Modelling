# Chef's Hat Gym — Opponent Modelling with DQN and PPO
**Coventry University | 7043SCN | Task 2 | Variant 0: Opponent Modelling** 

**Student ID: 12224702**

**Author: Mohamed Bseikri**

## Overview
This project investigates opponent modelling in the [Chef's Hat card game](https://github.com/pablovin/ChefsHatGYM) by training reinforcement learning agents against different opponent types and measuring the impact on learned policy quality.

Two RL algorithms are compared:

- DQN (Deep Q-Network) — value-based, off-policy
- PPO (Proximal Policy Optimisation) — policy gradient, on-policy

Each algorithm is trained under three opponent conditions (6 experiments in total), all for 44,000 games to ensure a fair comparison.

## Experiments & Results

| Exp | Algorithm | Opponent Type | Games | Avg Perf | Final 20% | Best |
|---|---|---|---|---|---|---|
| exp1 | DQN | vs Random     | 44,000 | 0.4997 | 0.5007 | 1.4781 |
| exp2 | DQN | vs Rule-based | 57,000 | 0.6322 | 0.6374 | 1.4832 |
| exp3 | DQN | vs Mixed      | 52,000 | 0.6446 | 0.6510 | 1.4742 |
| exp4 | PPO | vs Random     | 44,000 | 0.6228 | 0.6071 | 1.5015 |
| exp5 | PPO | vs Rule-based | 44,000 | 0.6086 | 0.5235 | 1.5409 |
| exp6 | PPO | vs Mixed      | 44,000 | 0.6359 | 0.6472 | 1.4637 |

## Results Plots

### DQN Learning Curves
![DQN Learning Curves](results/plots/dqn_learning_curves.png)

### PPO Learning Curves
![PPO Learning Curves](results/plots/ppo_learning_curves.png)

### DQN vs PPO by Opponent Type
![DQN vs PPO](results/plots/dqn_vs_ppo_curves.png)

### Final Performance Comparison (All 6 Experiments)
![Final Comparison](results/plots/final_comparison.png)

### Prerequisites
- python=3.10.19
- torch==2.10.0
- gymnasium==1.2.3
- chefshatgym==3.0.0.1


## Agent Architecture & Hyperparameters

### DQN

| Hyperparameter | Value |
|---|---|
| Network | 200 → 256 → 256 → 256 → 200 |
| Learning rate | 1e-3 → 1e-5 (CosineAnnealing) |
| Gamma | 0.99 |
| Epsilon | 1.0 → 0.05 (decay 0.9999) |
| Batch size | 128 |
| Replay buffer | 20,000 |
| Target network update | Every 10 games |
| Hidden size | 256 |

- Action masking: invalid actions zeroed out before argmax
- Experience replay with uniform sampling
- Target network for stable Q-value estimates

### PPO

| Hyperparameter | Value |
|---|---|
| Network (shared trunk) | 200 → 256 → 256 → 256 |
| Policy head | 256 → 200 |
| Value head | 256 → 1 |
| Learning rate | 3e-4 → 1e-5 (CosineAnnealing) |
| Gamma | 0.99 |
| GAE lambda | 0.95 |
| Clip epsilon | 0.2 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| PPO epochs per update | 4 |
| Batch size | 64 |
| Update frequency | Every 10 games |

- Action masking: logits + (valid_mask - 1) * 1e9 before Categorical sampling
- GAE for advantage estimation
- Entropy bonus to encourage exploration

---

## Environment

- **Game:** Chef's Hat (ChefsHatGym v3.0.0.1)
- **Action space:** 200-dimensional (discrete)
- **Players:** 4 (1 learning agent + 3 opponents)
- **Matches per game:** 5
- **Reward signal:** Game performance score

---

## References

- Pires, P. et al. (2023). *Chef's Hat Card Game for Equitable Interaction in Human-Robot Teams*. ChefsHatGym GitHub.
- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
