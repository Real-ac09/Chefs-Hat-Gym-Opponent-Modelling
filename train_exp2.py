"""
Experiment 2: DQN vs 3x Rule-Based Agents
Variant 0: Opponent Modelling — Student ID: 12224702

Run: python3 train_exp2.py
"""

from agents.rule_based_agent import RuleBasedAgent
from train_utils import run_experiment

def rulebased_opponents(game_idx):
    return [RuleBasedAgent(name=f"Rule{i}") for i in range(3)]

if __name__ == "__main__":
    log, dqn = run_experiment(
        exp_name="exp2_vs_rulebased",
        opponent_factory=rulebased_opponents,
    )
