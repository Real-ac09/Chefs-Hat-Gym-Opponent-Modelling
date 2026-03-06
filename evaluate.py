"""
Evaluation & Plotting Script — Variant 0: Opponent Modelling
Student ID: 12224702

Compares DQN vs PPO across all 3 opponent types (6 experiments total).
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOGS_DIR   = "results/logs"
PLOTS_DIR  = "results/plots"
MODELS_DIR = "results/models"

os.makedirs(PLOTS_DIR, exist_ok=True)

EVAL_GAMES = 44000

# DQN experiments
DQN_EXPS = {
    "exp1_vs_random":    {"label": "DQN vs Random",     "color": "#2196F3", "ckpt": "exp1_vs_random_ckpt_44000.pth"},
    "exp2_vs_rulebased": {"label": "DQN vs Rule-based", "color": "#F44336", "ckpt": "exp2_vs_rulebased_ckpt_44000.pth"},
    "exp3_vs_mixed":     {"label": "DQN vs Mixed",      "color": "#4CAF50", "ckpt": "exp3_vs_mixed_ckpt_44000.pth"},
}

# PPO experiments
PPO_EXPS = {
    "exp4_ppo_vs_random":    {"label": "PPO vs Random",     "color": "#1565C0", "ckpt": "exp4_ppo_vs_random_final.pth"},
    "exp5_ppo_vs_rulebased": {"label": "PPO vs Rule-based", "color": "#B71C1C", "ckpt": "exp5_ppo_vs_rulebased_final.pth"},
    "exp6_ppo_vs_mixed":     {"label": "PPO vs Mixed",      "color": "#1B5E20", "ckpt": "exp6_ppo_vs_mixed_final.pth"},
}

ALL_EXPS = {**DQN_EXPS, **PPO_EXPS}

# ---------------------------------------------------------------------------
# Load logs
# ---------------------------------------------------------------------------

def load_log(exp_name):
    path = os.path.join(LOGS_DIR, f"{exp_name}_log.json")
    if not os.path.exists(path):
        print(f"  Warning: {path} not found.")
        return None
    with open(path) as f:
        log = json.load(f)
    for key in ["performance_scores", "game_scores", "epsilon", "losses", "learning_rates"]:
        if key in log and isinstance(log[key], list):
            log[key] = log[key][:EVAL_GAMES]
    return log


def smooth(values, window=None):
    if window is None:
        window = max(50, len(values) // 100)
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window)/window, mode="valid").tolist()


def tail_mean(scores, frac=0.2):
    tail = scores[max(0, len(scores) - max(1, int(len(scores) * frac))):]
    return np.mean(tail), np.std(tail)


# ---------------------------------------------------------------------------
# Plot 1: DQN learning curves
# ---------------------------------------------------------------------------

def plot_dqn_curves(logs, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for exp_name, meta in DQN_EXPS.items():
        log = logs.get(exp_name)
        if log is None:
            continue
        scores   = log["performance_scores"]
        smoothed = smooth(scores)
        ax.plot(range(1, len(scores)+1), scores, alpha=0.12, color=meta["color"])
        ax.plot(range(1, len(smoothed)+1), smoothed, color=meta["color"],
                label=meta["label"], linewidth=2)
    ax.set_xlabel("Game", fontsize=12)
    ax.set_ylabel("Performance Score", fontsize=12)
    ax.set_title(f"DQN Learning Curves — {EVAL_GAMES:,} games", fontsize=14)
    ax.legend(fontsize=11); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: PPO learning curves
# ---------------------------------------------------------------------------

def plot_ppo_curves(logs, save_path):
    fig, ax = plt.subplots(figsize=(12, 5))
    for exp_name, meta in PPO_EXPS.items():
        log = logs.get(exp_name)
        if log is None:
            continue
        scores   = log["performance_scores"]
        smoothed = smooth(scores)
        ax.plot(range(1, len(scores)+1), scores, alpha=0.12, color=meta["color"])
        ax.plot(range(1, len(smoothed)+1), smoothed, color=meta["color"],
                label=meta["label"], linewidth=2)
    ax.set_xlabel("Game", fontsize=12)
    ax.set_ylabel("Performance Score", fontsize=12)
    ax.set_title(f"PPO Learning Curves — {EVAL_GAMES:,} games", fontsize=14)
    ax.legend(fontsize=11); ax.set_ylim(0, 1.1); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 3: DQN vs PPO side-by-side per opponent type
# ---------------------------------------------------------------------------

def plot_dqn_vs_ppo(logs, save_path):
    opponent_types = ["random", "rulebased", "mixed"]
    labels         = ["vs Random", "vs Rule-based", "vs Mixed"]
    dqn_keys = ["exp1_vs_random", "exp2_vs_rulebased", "exp3_vs_mixed"]
    ppo_keys = ["exp4_ppo_vs_random", "exp5_ppo_vs_rulebased", "exp6_ppo_vs_mixed"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, opp_label, dqn_key, ppo_key in zip(axes, labels, dqn_keys, ppo_keys):
        dqn_log = logs.get(dqn_key)
        ppo_log = logs.get(ppo_key)

        if dqn_log:
            s = smooth(dqn_log["performance_scores"])
            ax.plot(range(1, len(s)+1), s, color="#2196F3", label="DQN", linewidth=2)
        if ppo_log:
            s = smooth(ppo_log["performance_scores"])
            ax.plot(range(1, len(s)+1), s, color="#FF6F00", label="PPO", linewidth=2, linestyle="--")

        ax.set_title(opp_label, fontsize=12)
        ax.set_xlabel("Game", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    axes[0].set_ylabel("Performance Score", fontsize=11)
    fig.suptitle("DQN vs PPO Learning Curves by Opponent Type", fontsize=14, y=1.02)
    plt.tight_layout(); plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 4: Final comparison bar chart (all 6)
# ---------------------------------------------------------------------------

def plot_final_comparison(logs, save_path):
    all_labels, all_means, all_stds, all_colors = [], [], [], []

    for exp_name, meta in ALL_EXPS.items():
        log = logs.get(exp_name)
        if log is None:
            continue
        m, s = tail_mean(log["performance_scores"])
        all_labels.append(meta["label"])
        all_means.append(m)
        all_stds.append(s)
        all_colors.append(meta["color"])

    fig, ax = plt.subplots(figsize=(13, 5))
    x    = range(len(all_labels))
    bars = ax.bar(x, all_means, yerr=all_stds, capsize=4,
                  color=all_colors, alpha=0.85, edgecolor="black")
    for bar, mean in zip(bars, all_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Divider between DQN and PPO
    ax.axvline(x=2.5, color="grey", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(1, 0.92, "DQN", ha="center", fontsize=12, color="grey", style="italic")
    ax.text(4, 0.92, "PPO", ha="center", fontsize=12, color="grey", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Avg Performance Score (last 20%)", fontsize=11)
    ax.set_title(f"Final Performance — DQN vs PPO, {EVAL_GAMES:,} games each", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 5: Epsilon decay (DQN only)
# ---------------------------------------------------------------------------

def plot_epsilon_decay(logs, save_path):
    fig, ax = plt.subplots(figsize=(12, 4))
    for exp_name, meta in DQN_EXPS.items():
        log = logs.get(exp_name)
        if log is None or "epsilon" not in log:
            continue
        epsilons = log["epsilon"]
        ax.plot(range(1, len(epsilons)+1), epsilons, color=meta["color"],
                label=meta["label"], linewidth=2)
    ax.set_xlabel("Game", fontsize=12)
    ax.set_ylabel("Epsilon", fontsize=12)
    ax.set_title("DQN Epsilon Decay During Training", fontsize=14)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Plot 6: Loss curves
# ---------------------------------------------------------------------------

def plot_loss_curves(logs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    for ax, exp_dict, title in zip(axes, [DQN_EXPS, PPO_EXPS], ["DQN Loss", "PPO Loss"]):
        for exp_name, meta in exp_dict.items():
            log = logs.get(exp_name)
            if log is None:
                continue
            losses   = log["losses"]
            smoothed = smooth(losses)
            ax.plot(range(1, len(smoothed)+1), smoothed, color=meta["color"],
                    label=meta["label"], linewidth=2)
        ax.set_xlabel("Game", fontsize=11)
        ax.set_ylabel("Loss", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Head-to-head: best DQN vs best PPO + random
# ---------------------------------------------------------------------------

def evaluate_head_to_head(matches=100):
    from ChefsHatGym.gameRooms.chefs_hat_room_local import ChefsHatRoomLocal
    from ChefsHatGym.agents.agent_random import AgentRandon
    from agents.dqn_agent import DQNAgent
    from agents.ppo_agent import PPOAgent

    # Best DQN: exp2 vs rulebased
    dqn_path = os.path.join(MODELS_DIR, "exp2_vs_rulebased_ckpt_44000.pth")
    # Best PPO: exp6 vs mixed
    ppo_path = os.path.join(MODELS_DIR, "exp6_ppo_vs_mixed_final.pth")

    agents = []
    if os.path.exists(dqn_path):
        agents.append(DQNAgent(name="DQN_best", load_model=dqn_path, training=False))
        print(f"  Loaded DQN: exp2_vs_rulebased_ckpt_44000.pth")
    if os.path.exists(ppo_path):
        agents.append(PPOAgent(name="PPO_best", load_model=ppo_path, training=False))
        print(f"  Loaded PPO: exp6_ppo_vs_mixed_final.pth")

    # Pad to 4 with random
    while len(agents) < 4:
        agents.append(AgentRandon(name=f"Random_{len(agents)}"))

    room = ChefsHatRoomLocal(
        room_name="eval_dqn_vs_ppo",
        game_type="MATCHES",
        stop_criteria=matches,
        verbose_console=False, verbose_log=False,
        game_verbose_console=False, game_verbose_log=False,
        save_dataset=False, log_directory=LOGS_DIR,
    )
    for agent in agents:
        room.add_player(agent)

    import io
    old_stdout = sys.stdout; sys.stdout = io.StringIO()
    try:
        game_info = room.start_new_game()
    finally:
        sys.stdout = old_stdout
    return game_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Evaluation & Plotting — DQN vs PPO")
    print(f"  All experiments at {EVAL_GAMES:,} games")
    print("="*60)

    all_exp_names = list(ALL_EXPS.keys())
    logs = {name: load_log(name) for name in all_exp_names}
    valid_logs = {k: v for k, v in logs.items() if v is not None}

    if not valid_logs:
        print("\n  No logs found. Run training first.")
        sys.exit(1)

    print(f"\n  Loaded {len(valid_logs)}/6 experiment logs")
    print("\n  Generating plots...")

    plot_dqn_curves(valid_logs,      os.path.join(PLOTS_DIR, "dqn_learning_curves.png"))
    plot_ppo_curves(valid_logs,      os.path.join(PLOTS_DIR, "ppo_learning_curves.png"))
    plot_dqn_vs_ppo(valid_logs,      os.path.join(PLOTS_DIR, "dqn_vs_ppo_curves.png"))
    plot_final_comparison(valid_logs, os.path.join(PLOTS_DIR, "final_comparison.png"))
    plot_epsilon_decay(valid_logs,   os.path.join(PLOTS_DIR, "epsilon_decay.png"))
    plot_loss_curves(valid_logs,     os.path.join(PLOTS_DIR, "loss_curves.png"))

    # Summary table
    print("\n  Summary:")
    print("  {:<8} {:<12} {:<12} {:<12} {:<8}".format("Algorithm","Opponent","Avg Perf","Final 20%","Best"))
    print("  " + "-"*55)
    for exp_name, meta in ALL_EXPS.items():
        log = valid_logs.get(exp_name)
        if log is None:
            continue
        scores = log["performance_scores"]
        m, _   = tail_mean(scores)
        algo   = "DQN" if "ppo" not in exp_name else "PPO"
        opp    = meta["label"].split("vs ")[-1]
        print("  {:<8} {:<12} {:<12.4f} {:<12.4f} {:<8.4f}".format(algo, opp, np.mean(scores), m, max(scores)))

    # Head-to-head: best DQN vs best PPO
    print("\n  Running head-to-head: best DQN vs best PPO (100 matches)...")
    game_info = evaluate_head_to_head(matches=100)
    if game_info:
        names  = game_info.get("Player_Names", [])
        perfs  = game_info.get("Game_Performance_Score", [])
        scores = game_info.get("Game_Score", [])
        print("\n  Head-to-head results:")
        print("  {:<30} {:<14} {}".format("Agent","Perf Score","Game Score"))
        print("  " + "-"*55)
        for n, p, s in zip(names, perfs, scores):
            print("  {:<30} {:<14.4f} {}".format(n, p, s))

    print(f"\n  All plots saved to {PLOTS_DIR}/")
