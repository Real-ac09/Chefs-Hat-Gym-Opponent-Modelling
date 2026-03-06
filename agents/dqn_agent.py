"""
DQN Agent for Chef's Hat Card Game
Variant 0: Opponent Modelling
Student ID: 12224702

Observation: 200-element array where 1s indicate valid actions.
Action: one-hot 200-element array with a 1 at the chosen action index.
"""

import numpy as np
import random
import os
from collections import deque

from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    raise ImportError("PyTorch not found. Run: pip install torch")


class DQNNetwork(nn.Module):
    def __init__(self, obs_size=200, action_size=200, hidden_size=256):
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent(ChefsHatPlayer):
    """
    Double DQN agent for Chef's Hat.
    """

    OBS_SIZE    = 200
    ACTION_SIZE = 200

    def __init__(
        self,
        name,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9999,
        batch_size=128,
        buffer_capacity=20000,
        target_update_freq=10,
        hidden_size=256,
        save_dir="results/models",
        load_model=None,
        training=True,
        verbose_console=False,
        verbose_log=False,
        log_directory="",
    ):
        super().__init__(
            agent_suffix="DQN",
            name=name,
            verbose_console=verbose_console,
            verbose_log=verbose_log,
            log_directory=log_directory,
            use_sufix=False,
        )

        self.training    = training
        self.save_dir    = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.gamma              = gamma
        self.epsilon            = epsilon_start if training else epsilon_end
        self.epsilon_end        = epsilon_end
        self.epsilon_decay      = epsilon_decay
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQNNetwork(self.OBS_SIZE, self.ACTION_SIZE, hidden_size).to(self.device)
        self.target_net = DQNNetwork(self.OBS_SIZE, self.ACTION_SIZE, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.episode_count   = 0
        self._last_state     = None
        self._last_action    = None

        self.episode_rewards = []
        self.episode_losses  = []
        self.win_history     = []
        self.epsilon_history = []

        if load_model and os.path.exists(load_model):
            self.load(load_model)
            print(f"[{name}] Loaded model from {load_model}")

    # -----------------------------------------------------------------------
    # ChefsHatPlayer interface
    # -----------------------------------------------------------------------

    def get_action(self, observation):
        observation = np.array(observation, dtype=np.float32)

        net_input = np.zeros(self.OBS_SIZE, dtype=np.float32)
        copy_len = min(len(observation), self.OBS_SIZE)
        net_input[:copy_len] = observation[:copy_len]

        valid_indices = [i for i in np.where(net_input == 1)[0].tolist()]
        if not valid_indices:
            valid_indices = [self.OBS_SIZE - 1]

        if self.training and random.random() < self.epsilon:
            chosen = random.choice(valid_indices)
        else:
            with torch.no_grad():
                obs_t  = torch.FloatTensor(net_input).unsqueeze(0).to(self.device)
                q_vals = self.policy_net(obs_t).squeeze(0).cpu().numpy()
            masked = np.full(self.OBS_SIZE, -1e9)
            for idx in valid_indices:
                masked[idx] = q_vals[idx]
            chosen = int(np.argmax(masked))
            if chosen not in valid_indices:
                chosen = random.choice(valid_indices)

        self._last_state  = net_input.copy()
        self._last_action = chosen

        action = np.zeros(self.OBS_SIZE)
        action[chosen] = 1
        return action

    def get_exhanged_cards(self, cards, num_cards):
        return list(cards[:num_cards])

    def get_reward(self, reward):
        if not self.training:
            return
        if self._last_state is not None and self._last_action is not None:
            next_state = np.zeros(self.OBS_SIZE, dtype=np.float32)
            self.replay_buffer.push(
                self._last_state, self._last_action,
                float(reward), next_state, 1.0
            )
            self.episode_rewards.append(float(reward))
            self._last_state  = None
            self._last_action = None
        loss = self._learn()
        if loss is not None:
            self.episode_losses.append(loss)

    def update_my_action(self, info):
        if not self.training:
            return
        if self._last_state is not None and self._last_action is not None:
            next_obs = info.get("Observation_After", None)
            if next_obs is not None and len(next_obs) == self.OBS_SIZE:
                next_state = np.array(next_obs, dtype=np.float32)
            else:
                next_state = np.zeros(self.OBS_SIZE, dtype=np.float32)
            self.replay_buffer.push(
                self._last_state, self._last_action,
                0.0, next_state, 0.0
            )

    def update_action_others(self, info):
        pass

    def update_start_match(self, cards, player_names, current_player):
        pass

    def update_end_match(self, info):
        self.episode_count += 1
        player_names = info.get("Player_Names", [])
        perf         = info.get("Game_Performance_Score", [])
        if self.name in player_names:
            idx = player_names.index(self.name)
            self.win_history.append(perf[idx] if idx < len(perf) else 0.0)
        if self.training:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.epsilon_history.append(self.epsilon)
        if self.episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_game_over(self):
        pass

    def update_exchange_cards(self, given_cards, received_cards):
        pass

    def observe_special_action(self, action, player):
        pass

    def do_special_action(self, info, action):
        return False

    def saveModelIn(self, path):
        self.save(path)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1)
            next_q       = self.target_net(next_states_t).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q     = rewards_t + self.gamma * next_q * (1 - dones_t)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def save(self, path=None, scheduler=None):
        if path is None:
            path = os.path.join(self.save_dir, f"{self.name}_dqn.pth")
        checkpoint = {
            "policy_net":      self.policy_net.state_dict(),
            "target_net":      self.target_net.state_dict(),
            "optimizer":       self.optimizer.state_dict(),
            "epsilon":         self.epsilon,
            "episode_count":   self.episode_count,
            "episode_rewards": self.episode_rewards,
            "episode_losses":  self.episode_losses,
            "win_history":     self.win_history,
            "epsilon_history": self.epsilon_history,
        }
        if scheduler is not None:
            checkpoint["scheduler"] = scheduler.state_dict()
        torch.save(checkpoint, path)

    def load(self, path, scheduler=None):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon         = checkpoint.get("epsilon", self.epsilon_end)
        self.episode_count   = checkpoint.get("episode_count", 0)
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_losses  = checkpoint.get("episode_losses", [])
        self.win_history     = checkpoint.get("win_history", [])
        self.epsilon_history = checkpoint.get("epsilon_history", [])
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
