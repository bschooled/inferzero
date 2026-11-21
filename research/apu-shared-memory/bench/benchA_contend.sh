#!/usr/bin/env bash
# Experiment A: does a CPU weight-stream lane add bandwidth to the GPU's, or contend?
cd ~/inferzero
M=~/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
P="Explain quicksort in one paragraph."
echo "=== baseline: CPU GEMV alone ==="
~/ov/bin/python ~/gemv_loop.py 20
sleep 40
echo "=== GPU tg (300 tok, ~100s streaming) + CPU GEMV concurrent ==="
( timeout 400 ./bin/runz "$M" --bench-pp 2048 --ctx 2128 --n 300 "$P" 2>&1 | grep -E "^(tg300|weights)" & )
sleep 62
~/ov/bin/python ~/gemv_loop.py 40
wait
