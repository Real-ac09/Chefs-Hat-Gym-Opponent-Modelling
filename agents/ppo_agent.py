"""
PPO Agent for Chef's Hat Card Game
Variant 0: Opponent Modelling - Student ID: 12224702
PPO-Clip with shared actor-critic network, action masking, GAE
"""

import numpy as np
import os

from ChefsHatGym.agents.base_classes.chefs_hat_player import ChefsHatPlayer

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
except ImportError:
    raise ImportError("PyTorch not found.")


class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_size=200, action_size=200, hidden_size=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        f = self.trunk(x)
        return self.policy_head(f), self.value_head(f).squeeze(-1)

    def get_action(self, obs, valid_mask):
        logits, value = self.forward(obs)
        logits = logits + (valid_mask.float() - 1) * 1e9
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, valid_mask, actions):
        logits, values = self.forward(obs)
        logits = logits + (valid_mask.float() - 1) * 1e9
        dist   = Categorical(logits=logits)
        return dist.log_prob(actions), values, dist.entropy()


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []; self.valid_masks = []; self.actions = []
        self.log_probs = []; self.rewards = []; self.values = []; self.dones = []

    def push(self, state, valid_mask, action, log_prob, reward, value, done):
        self.states.append(state);    self.valid_masks.append(valid_mask)
        self.actions.append(action);  self.log_probs.append(log_prob)
        self.rewards.append(reward);  self.values.append(value)
        self.dones.append(done)

    def __len__(self):
        return len(self.states)


class PPOAgent(ChefsHatPlayer):
    OBS_SIZE    = 200
    ACTION_SIZE = 200

    def __init__(self, name, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5,
                 ppo_epochs=4, batch_size=64, update_freq=10, hidden_size=256,
                 save_dir="results/models", load_model=None, training=True,
                 verbose_console=False, verbose_log=False, log_directory=""):
        super().__init__(agent_suffix="PPO", name=name, verbose_console=verbose_console,
                         verbose_log=verbose_log, log_directory=log_directory, use_sufix=False)

        self.training     = training
        self.save_dir     = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef   = value_coef
        self.ppo_epochs   = ppo_epochs
        self.batch_size   = batch_size
        self.update_freq  = update_freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net       = ActorCriticNetwork(self.OBS_SIZE, self.ACTION_SIZE, hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        self.buffer        = RolloutBuffer()
        self.episode_count = 0

        self._last_state      = None
        self._last_action     = None
        self._last_log_prob   = None
        self._last_value      = None
        self._last_valid_mask = None

        self.episode_losses = []
        self.win_history    = []
        self.policy_losses  = []
        self.value_losses   = []
        self.entropies      = []

        if load_model and os.path.exists(load_model):
            self.load(load_model)
            print(f"[{name}] Loaded model from {load_model}")

    def get_action(self, observation):
        observation = np.array(observation, dtype=np.float32)
        net_input   = np.zeros(self.OBS_SIZE, dtype=np.float32)
        net_input[:min(len(observation), self.OBS_SIZE)] = observation[:self.OBS_SIZE]

        valid_mask = torch.BoolTensor(net_input.astype(bool)).to(self.device)
        if not valid_mask.any():
            valid_mask[-1] = True

        obs_t  = torch.FloatTensor(net_input).unsqueeze(0).to(self.device)
        mask_t = valid_mask.unsqueeze(0)

        if self.training:
            action_t, log_prob_t, value_t = self.net.get_action(obs_t, mask_t)
            self._last_state      = net_input.copy()
            self._last_action     = action_t.item()
            self._last_log_prob   = log_prob_t.item()
            self._last_value      = value_t.item()
            self._last_valid_mask = net_input.astype(bool).copy()
        else:
            with torch.no_grad():
                logits, _ = self.net(obs_t)
                logits    = logits + (mask_t.float() - 1) * 1e9
                self._last_action = logits.argmax(dim=-1).item()

        action_one_hot = np.zeros(self.ACTION_SIZE)
        action_one_hot[self._last_action] = 1
        return action_one_hot

    def get_exhanged_cards(self, cards, num_cards):
        return list(cards[:num_cards])

    def get_reward(self, reward):
        if not self.training or self._last_state is None:
            return
        self.buffer.push(self._last_state, self._last_valid_mask, self._last_action,
                         self._last_log_prob, float(reward), self._last_value, done=True)
        self._last_state = None

    def update_my_action(self, info):
        if not self.training or self._last_state is None:
            return
        self.buffer.push(self._last_state, self._last_valid_mask, self._last_action,
                         self._last_log_prob, 0.0, self._last_value, done=False)
        self._last_state = None

    def update_action_others(self, info): pass
    def update_start_match(self, cards, player_names, current_player): pass

    def update_end_match(self, info):
        self.episode_count += 1
        player_names = info.get("Player_Names", [])
        perf         = info.get("Game_Performance_Score", [])
        if self.name in player_names:
            idx = player_names.index(self.name)
            self.win_history.append(perf[idx] if idx < len(perf) else 0.0)

        if self.training and self.episode_count % self.update_freq == 0 and len(self.buffer) > 0:
            metrics = self._ppo_update()
            if metrics:
                self.policy_losses.append(metrics["policy_loss"])
                self.value_losses.append(metrics["value_loss"])
                self.entropies.append(metrics["entropy"])
                self.episode_losses.append(metrics["policy_loss"] + metrics["value_loss"])

    def update_game_over(self): pass
    def update_exchange_cards(self, given_cards, received_cards): pass
    def observe_special_action(self, action, player): pass
    def do_special_action(self, info, action): return False
    def saveModelIn(self, path): self.save(path)

    def _compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        next_value = 0.0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + self.gamma * next_value * (1 - d) - v
            gae   = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            advantages.insert(0, gae)
            next_value = v
        returns = [a + v for a, v in zip(advantages, values)]
        return advantages, returns

    def _ppo_update(self):
        if len(self.buffer) == 0:
            return None

        states      = np.array(self.buffer.states,      dtype=np.float32)
        valid_masks = np.array(self.buffer.valid_masks, dtype=bool)
        actions     = np.array(self.buffer.actions,     dtype=np.int64)
        old_lps     = np.array(self.buffer.log_probs,   dtype=np.float32)

        advantages, returns = self._compute_gae(
            self.buffer.rewards, self.buffer.values, self.buffer.dones)

        advantages = np.array(advantages, dtype=np.float32)
        returns    = np.array(returns,    dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t  = torch.FloatTensor(states).to(self.device)
        masks_t   = torch.BoolTensor(valid_masks).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_lps_t = torch.FloatTensor(old_lps).to(self.device)
        adv_t     = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        n = len(states)
        total_pl = total_vl = total_ent = count = 0

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                log_probs, values_pred, entropy = self.net.evaluate(
                    states_t[idx], masks_t[idx], actions_t[idx])

                ratio  = torch.exp(log_probs - old_lps_t[idx])
                surr1  = ratio * adv_t[idx]
                surr2  = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_t[idx]
                pl     = -torch.min(surr1, surr2).mean()
                vl     = nn.MSELoss()(values_pred, returns_t[idx])
                el     = -entropy.mean()
                loss   = pl + self.value_coef * vl + self.entropy_coef * el

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()

                total_pl  += pl.item()
                total_vl  += vl.item()
                total_ent += entropy.mean().item()
                count     += 1

        self.buffer.clear()
        return {"policy_loss": total_pl / max(1, count),
                "value_loss":  total_vl / max(1, count),
                "entropy":     total_ent / max(1, count)}

    def save(self, path=None, scheduler=None):
        if path is None:
            path = os.path.join(self.save_dir, f"{self.name}_ppo.pth")
        ckpt = {"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict(),
                "episode_count": self.episode_count, "win_history": self.win_history,
                "episode_losses": self.episode_losses, "policy_losses": self.policy_losses,
                "value_losses": self.value_losses, "entropies": self.entropies}
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()
        torch.save(ckpt, path)

    def load(self, path, scheduler=None):
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.episode_count  = ckpt.get("episode_count", 0)
        self.win_history    = ckpt.get("win_history",   [])
        self.episode_losses = ckpt.get("episode_losses",[])
        self.policy_losses  = ckpt.get("policy_losses", [])
        self.value_losses   = ckpt.get("value_losses",  [])
        self.entropies      = ckpt.get("entropies",     [])
        if scheduler is not None and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
