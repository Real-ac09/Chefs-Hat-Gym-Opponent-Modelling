"""
PPO Training Script - Exp 1: vs Random opponents
Variant 0: Opponent Modelling - Student ID: 12224702
"""

import os, sys, json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from ChefsHatGym.agents.agent_random import AgentRandon
from agents.ppo_agent import PPOAgent

RESULTS_DIR = "results"
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

NUM_GAMES        = 44000
MATCHES_PER_GAME = 5
EXP_NAME         = "exp5_ppo_vs_rulebased"
PRINT_EVERY      = 500
CKPT_EVERY       = 1000

PPO_PARAMS = dict(
    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
    clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5,
    ppo_epochs=4, batch_size=64, update_freq=10, hidden_size=256,
)

def make_opponents():
    try:
        from ChefsHatGym.agents.agent_rule_based import AgentRuleBased
        return [AgentRuleBased(name=f"RuleBased_{i}") for i in range(3)]
    except ImportError:
        try:
            from ChefsHatGym.agents.agent_rule_based_hard import AgentRuleBasedHard
            return [AgentRuleBasedHard(name=f"RuleBased_{i}") for i in range(3)]
        except ImportError:
            return [AgentRandon(name=f"Random_{i}") for i in range(3)]

def run_game(ppo_agent, opponents, game_name):
    room = ChefsHatRoomLocal(room_name=game_name, game_type="MATCHES",
        stop_criteria=MATCHES_PER_GAME, verbose_console=False, verbose_log=False,
        game_verbose_console=False, game_verbose_log=False,
        save_dataset=False, log_directory=LOGS_DIR)
    room.add_player(ppo_agent)
    for opp in opponents:
        room.add_player(opp)
    import io
    old_stdout = sys.stdout; sys.stdout = io.StringIO()
    try:
        game_info = room.start_new_game()
    finally:
        sys.stdout = old_stdout
    return game_info

def get_agent_perf(game_info, agent_name):
    names  = game_info.get("Player_Names", [])
    perfs  = game_info.get("Game_Performance_Score", [])
    scores = game_info.get("Game_Score", [])
    if agent_name in names:
        idx = names.index(agent_name)
        return (perfs[idx] if idx < len(perfs) else 0.0,
                scores[idx] if idx < len(scores) else 0)
    return 0.0, 0

def find_latest_checkpoint():
    if not os.path.exists(MODELS_DIR):
        return None, 0
    ckpts = [f for f in os.listdir(MODELS_DIR)
             if f.startswith(EXP_NAME + "_ckpt_") and f.endswith(".pth")]
    if not ckpts:
        return None, 0
    nums = []
    for f in ckpts:
        try:
            num = int(f.replace(EXP_NAME + "_ckpt_", "").replace(".pth", ""))
            nums.append((num, f))
        except ValueError:
            pass
    if not nums:
        return None, 0
    latest_num, latest_file = max(nums, key=lambda x: x[0])
    return os.path.join(MODELS_DIR, latest_file), latest_num

if __name__ == "__main__":
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    print(f"\n{'='*60}\n  PPO Exp: {EXP_NAME}\n  {NUM_GAMES:,} games\n{'='*60}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    ppo = PPOAgent(name="PPOPlayer", save_dir=MODELS_DIR, training=True, **PPO_PARAMS)
    scheduler = CosineAnnealingLR(ppo.optimizer, T_max=NUM_GAMES, eta_min=1e-5)

    ckpt_path, start_game = find_latest_checkpoint()
    log_path = os.path.join(LOGS_DIR, f"{EXP_NAME}_log.json")

    if ckpt_path and os.path.exists(log_path):
        ppo.load(ckpt_path, scheduler=scheduler)
        with open(log_path) as f:
            log = json.load(f)
        for key in ["performance_scores","game_scores","losses","policy_losses","value_losses","entropies","learning_rates"]:
            if key in log:
                log[key] = log[key][:start_game]
        print(f"  Resumed from: {ckpt_path} (game {start_game:,})")
    else:
        start_game = 0
        log = {"experiment": EXP_NAME, "performance_scores": [], "game_scores": [],
               "losses": [], "policy_losses": [], "value_losses": [],
               "entropies": [], "learning_rates": [], "win_history": []}
        print(f"  Starting fresh.")

    for game_idx in range(start_game, NUM_GAMES):
        game_info = run_game(ppo, make_opponents(), f"{EXP_NAME}_g{game_idx}")
        perf, score = get_agent_perf(game_info, "PPOPlayer")
        avg_loss   = float(np.mean(ppo.episode_losses[-50:])) if ppo.episode_losses else 0.0
        current_lr = scheduler.get_last_lr()[0] if game_idx > 0 else PPO_PARAMS["learning_rate"]

        log["performance_scores"].append(perf)
        log["game_scores"].append(score)
        log["losses"].append(avg_loss)
        log["policy_losses"].append(ppo.policy_losses[-1] if ppo.policy_losses else 0.0)
        log["value_losses"].append(ppo.value_losses[-1]   if ppo.value_losses  else 0.0)
        log["entropies"].append(ppo.entropies[-1]         if ppo.entropies     else 0.0)
        log["learning_rates"].append(current_lr)
        log["win_history"] = ppo.win_history
        scheduler.step()

        if (game_idx + 1) % PRINT_EVERY == 0 or game_idx == start_game:
            recent_perf = np.mean(log["performance_scores"][-PRINT_EVERY:])
            print(f"  Game {game_idx+1:7d}/{NUM_GAMES:,} | Perf: {perf:.3f} | "
                  f"Avg{PRINT_EVERY}: {recent_perf:.3f} | LR: {current_lr:.2e} | Loss: {avg_loss:.4f}")

        if (game_idx + 1) % CKPT_EVERY == 0:
            ckpt = os.path.join(MODELS_DIR, f"{EXP_NAME}_ckpt_{game_idx+1}.pth")
            ppo.save(ckpt, scheduler=scheduler)
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)
            print(f"  [Checkpoint @ game {game_idx+1:,}]")

    ppo.save(os.path.join(MODELS_DIR, f"{EXP_NAME}_final.pth"), scheduler=scheduler)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Done! Avg: {np.mean(log['performance_scores']):.4f} | "
          f"Final 1K: {np.mean(log['performance_scores'][-1000:]):.4f}")
