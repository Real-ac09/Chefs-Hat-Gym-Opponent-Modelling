#!/bin/bash
# Launch all 3 experiments in parallel
# Logs go to results/logs/expN.out
# Usage: bash train_all_parallel.sh

mkdir -p results/logs

echo "Launching all 3 experiments in parallel..."
echo "Monitor progress with: tail -f results/logs/exp1.out"
echo ""

python3 train_exp1.py > results/logs/exp1.out 2>&1 &
PID1=$!
echo "Exp 1 (vs Random)    started — PID $PID1"

python3 train_exp2.py > results/logs/exp2.out 2>&1 &
PID2=$!
echo "Exp 2 (vs Rule-based) started — PID $PID2"

python3 train_exp3.py > results/logs/exp3.out 2>&1 &
PID3=$!
echo "Exp 3 (vs Mixed)     started — PID $PID3"

echo ""
echo "All running! Monitor with:"
echo "  tail -f results/logs/exp1.out"
echo "  tail -f results/logs/exp2.out"
echo "  tail -f results/logs/exp3.out"
echo ""
echo "Or watch all at once:"
echo "  tail -f results/logs/exp*.out"
echo ""

# Wait for all to finish
wait $PID1 $PID2 $PID3
echo "All experiments complete!"
python3 evaluate.py
