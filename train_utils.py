"""
Shared training utilities for all experiments.
Variant 0: Opponent Modelling — Student ID: 12224702
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
from agents.dqn_agent import DQNAgent

RESULTS_DIR = "results"
MODELS_DIR  = os.path.join(RESULTS_DIR, "models")
LOGS_DIR    = os.path.join(RESULTS_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_GAMES        = 500000
MATCHES_PER_GAME = 5

DQN_PARAMS = dict(
    learning_rate      = 1e-3,          # CosineAnnealing decays to 1e-5
    gamma              = 0.99,
    epsilon_start      = 1.0,
    epsilon_end        = 0.05,
    epsilon_decay      = 0.9999,        # hits ~0.05 around game 46000
    batch_size         = 128,
    buffer_capacity    = 20000,
    target_update_freq = 10,
    hidden_size        = 256,
)


def run_game(dqn_agent, opponent_agents, game_name, match_count):
    room = ChefsHatRoomLocal(
        room_name=game_name,
        game_type="MATCHES",
        stop_criteria=match_count,
        verbose_console=False,
        verbose_log=False,
        game_verbose_console=False,
        game_verbose_log=False,
        save_dataset=False,
        log_directory=LOGS_DIR,
    )
    room.add_player(dqn_agent)
    for opp in opponent_agents:
        room.add_player(opp)

    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
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
        return (
            perfs[idx]  if idx < len(perfs)  else 0.0,
            scores[idx] if idx < len(scores) else 0
        )
    return 0.0, 0


def run_experiment(exp_name, opponent_factory, num_games=NUM_GAMES, matches_per_game=MATCHES_PER_GAME):
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_name}")
    print(f"  {num_games:,} games x {matches_per_game} matches each")
    print(f"  Total matches: {num_games * matches_per_game:,}")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    dqn = DQNAgent(
        name="DQNPlayer",
        save_dir=MODELS_DIR,
        training=True,
        **DQN_PARAMS,
    )

    # CosineAnnealingLR: lr 1e-3 → 1e-5 over full run
    scheduler = CosineAnnealingLR(dqn.optimizer, T_max=num_games, eta_min=1e-5)

    log = {
        "experiment":         exp_name,
        "performance_scores": [],
        "game_scores":        [],
        "epsilon":            [],
        "losses":             [],
        "learning_rates":     [],
        "win_history":        [],
    }

    PRINT_EVERY      = 500
    CHECKPOINT_EVERY = 1000

    for game_idx in range(num_games):
        opponents = opponent_factory(game_idx)
        game_name = f"{exp_name}_g{game_idx}"
        game_info = run_game(dqn, opponents, game_name, matches_per_game)

        perf, score = get_agent_perf(game_info, "DQNPlayer")
        avg_loss    = float(np.mean(dqn.episode_losses[-50:])) if dqn.episode_losses else 0.0
        current_lr  = scheduler.get_last_lr()[0] if game_idx > 0 else DQN_PARAMS["learning_rate"]

        log["performance_scores"].append(perf)
        log["game_scores"].append(score)
        log["epsilon"].append(dqn.epsilon)
        log["losses"].append(avg_loss)
        log["learning_rates"].append(current_lr)
        log["win_history"] = dqn.win_history

        scheduler.step()

        if (game_idx + 1) % PRINT_EVERY == 0 or game_idx == 0:
            recent_perf = np.mean(log["performance_scores"][-PRINT_EVERY:])
            print(f"  Game {game_idx+1:7d}/{num_games:,} | "
                  f"Perf: {perf:.3f} | "
                  f"Avg{PRINT_EVERY}: {recent_perf:.3f} | "
                  f"Eps: {dqn.epsilon:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Loss: {avg_loss:.4f}")

        if (game_idx + 1) % CHECKPOINT_EVERY == 0:
            ckpt = os.path.join(MODELS_DIR, f"{exp_name}_ckpt_{game_idx+1}.pth")
            dqn.save(ckpt, scheduler=scheduler)
            log_path = os.path.join(LOGS_DIR, f"{exp_name}_log.json")
            with open(log_path, "w") as f:
                json.dump(log, f, indent=2)
            print(f"  [Checkpoint @ game {game_idx+1:,} saved]")

    model_path = os.path.join(MODELS_DIR, f"{exp_name}_final.pth")
    dqn.save(model_path, scheduler=scheduler)

    log_path = os.path.join(LOGS_DIR, f"{exp_name}_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    avg_perf   = np.mean(log["performance_scores"])
    final_perf = np.mean(log["performance_scores"][-1000:])
    print(f"\n  Done!")
    print(f"  Overall avg performance    : {avg_perf:.4f}")
    print(f"  Final 1000 games avg       : {final_perf:.4f}")
    print(f"  Model: {model_path}")

    return log, dqn
