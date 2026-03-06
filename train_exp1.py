"""
Experiment 1: DQN vs 3x Random Agents
Variant 0: Opponent Modelling — Student ID: 12224702

Run: python3 train_exp1.py
"""

from ChefsHatGym.agents.agent_random import AgentRandon
from train_utils import run_experiment

def random_opponents(game_idx):
    return [AgentRandon(name=f"Rnd{i}") for i in range(3)]

if __name__ == "__main__":
    log, dqn = run_experiment(
        exp_name="exp1_vs_random",
        opponent_factory=random_opponents,
    )
