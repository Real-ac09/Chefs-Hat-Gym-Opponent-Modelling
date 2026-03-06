"""
Rule-Based Agent for Chef's Hat Card Game
Variant 0: Opponent Modelling — used as a stronger opponent than random.
Strategy: always play the lowest-index valid action.
"""

import numpy as np
from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer


class RuleBasedAgent(ChefsHatPlayer):

    def __init__(self, name, verbose_console=False, verbose_log=False, log_directory=""):
        super().__init__(
            agent_suffix="RuleBased",
            name=name,
            verbose_console=verbose_console,
            verbose_log=verbose_log,
            log_directory=log_directory,
            use_sufix=False,
        )

    def get_action(self, observation):
        """
        Always pick the lowest-index valid action.
        Returns a 200-element one-hot array.
        """
        observation = np.array(observation, dtype=np.float32)

        # Work within 200 elements always
        obs = np.zeros(200, dtype=np.float32)
        copy_len = min(len(observation), 200)
        obs[:copy_len] = observation[:copy_len]

        valid_indices = np.where(obs == 1)[0].tolist()
        if not valid_indices:
            valid_indices = [199]  # fallback: pass

        chosen = min(valid_indices)  # lowest index = most conservative play

        action = np.zeros(200)
        action[chosen] = 1
        return action

    def get_exhanged_cards(self, cards, num_cards):
        return list(cards[:num_cards])

    def get_reward(self, reward):
        pass

    def update_my_action(self, info):
        pass

    def update_action_others(self, info):
        pass

    def update_start_match(self, cards, player_names, current_player):
        pass

    def update_end_match(self, info):
        pass

    def update_game_over(self):
        pass

    def update_exchange_cards(self, given_cards, received_cards):
        pass

    def observe_special_action(self, action, player):
        pass

    def do_special_action(self, info, action):
        return False

    def saveModelIn(self, path):
        pass
