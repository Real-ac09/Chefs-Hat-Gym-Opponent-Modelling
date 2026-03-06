"""
Experiment 3: DQN vs Mixed Opponents (1x Random + 2x Rule-based)
Variant 0: Opponent Modelling — Student ID: 12224702

Run: python3 train_exp3.py
"""

from ChefsHatGym.agents.agent_random import AgentRandon
from agents.rule_based_agent import RuleBasedAgent
from train_utils import run_experiment

def mixed_opponents(game_idx):
    return [
        AgentRandon(name="RndOpp"),
        RuleBasedAgent(name="Rule0"),
        RuleBasedAgent(name="Rule1"),
    ]

if __name__ == "__main__":
    log, dqn = run_experiment(
        exp_name="exp3_vs_mixed",
        opponent_factory=mixed_opponents,
    )
