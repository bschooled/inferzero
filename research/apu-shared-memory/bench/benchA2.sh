#!/usr/bin/env bash
# Experiment A2 (clean): single CPU lane alone; GPU tg + one CPU lane; GPU tg + two CPU lanes.
cd ~/inferzero
M=~/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
P="Explain quicksort in one paragraph."
g() { ~/ov/bin/python ~/gemv_loop.py "$1" "$2"; }
echo "=== A2-1: one CPU GEMV lane alone (20s) ==="
g 20 "solo1"
sleep 30
echo "=== A2-2: GPU tg alone (n=200) ==="
( timeout 400 ./bin/runz "$M" --bench-pp 2048 --ctx 2128 --n 200 "$P" 2>&1 | grep -E "^(tg200|weights)" & )
sleep 95
echo "=== A2-3: GPU tg + ONE CPU lane ==="
( timeout 400 ./bin/runz "$M" --bench-pp 2048 --ctx 2128 --n 200 "$P" 2>&1 | grep -E "^(tg200|weights)" & )
sleep 60
g 25 "plus1lane" &
wait
sleep 20
echo "=== A2-4: GPU tg + TWO CPU lanes ==="
( timeout 400 ./bin/runz "$M" --bench-pp 2048 --ctx 2128 --n 260 "$P" 2>&1 | grep -E "^(tg260|weights)" & )
sleep 60
g 30 "laneA" &
g 30 "laneB" &
wait
