#!/usr/bin/env bash
# Round 2: prompt-path tuning + NPU + peak rates. 30s cooldowns; engine does 3 pp runs each.
cd ~/inferzero
M=~/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
run() { sleep 30; echo "=== $* ==="; timeout 900 ./bin/runz "$M" "$@" 2>&1 | grep -E "^(loaded|pp[0-9]+|tg[0-9]+|tg steps|weights|WARN|ERROR|error|npu|NPU)"; }
run --bench-pp 8192 --ctx 8192 --n 16
run --bench-pp 8192 --ctx 8192 --chunk 2048 --n 16
run --bench-pp 8192 --ctx 8192 --npu force --n 16
echo "=== gpu_bench ==="; timeout 300 ./bin/gpu_bench 2>&1 | tail -15
