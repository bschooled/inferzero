#!/usr/bin/env bash
# Round 5: NPU retest with the new driver (working geometry) + context ceiling ladder.
cd ~/inferzero
M=~/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
P="Explain quicksort in one paragraph."
run() { sleep 45; echo "=== $* ==="; timeout 900 ./bin/runz "$M" "$@" "$P" 2>&1 | grep -E "^(loaded|pp[0-9]+|tg[0-9]+|weights|Level Zero|GPU memory|WARN|ERROR|error|ARC_NPU|npu|NPU)" ; echo "rc=$?"; }
run --bench-pp 2048 --npu force --n 16
run --bench-pp 4096 --ctx 4096 --n 32
run --bench-pp 8192 --ctx 8224 --chunk 1024 --n 32
run --bench-pp 2048 --ctx 2128 --n 32
