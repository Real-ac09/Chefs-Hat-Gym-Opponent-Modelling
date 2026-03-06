#!/bin/bash
# Launch all 3 PPO experiments in parallel
# Auto-resumes from latest checkpoint if available

cd "$(dirname "$0")"

mkdir -p results/logs

echo "Launching all 3 PPO experiments in parallel..."

python3 train_ppo_exp1.py > results/logs/exp4_ppo.out 2>&1 &
PPO1_PID=$!
echo "PPO Exp1 (vs Random)     started — PID $PPO1_PID"

python3 train_ppo_exp2.py > results/logs/exp5_ppo.out 2>&1 &
PPO2_PID=$!
echo "PPO Exp2 (vs Rule-based) started — PID $PPO2_PID"

python3 train_ppo_exp3.py > results/logs/exp6_ppo.out 2>&1 &
PPO3_PID=$!
echo "PPO Exp3 (vs Mixed)      started — PID $PPO3_PID"

echo ""
echo "Monitor with:"
echo "  grep 'Game' results/logs/exp4_ppo.out | tail -3"
echo "  grep 'Game' results/logs/exp5_ppo.out | tail -3"
echo "  grep 'Game' results/logs/exp6_ppo.out | tail -3"
echo ""
echo "Kill all with:"
echo "  kill $PPO1_PID $PPO2_PID $PPO3_PID"

wait $PPO1_PID $PPO2_PID $PPO3_PID
echo "All PPO experiments complete!"
