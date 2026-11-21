#!/usr/bin/env bash
# Round 3 (fixed): prompt path at 8192 + chunk sweep + NPU force + MTP acceptance line.
cd ~/inferzero
M=~/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
P="Explain quicksort in one paragraph."
run() { sleep 30; echo "=== $* ==="; timeout 900 ./bin/runz "$M" "$@" "$P" 2>&1 | tail -14; }
run --bench-pp 8192 --ctx 8192 --n 16
run --bench-pp 8192 --ctx 8192 --chunk 2048 --n 16
run --bench-pp 8192 --ctx 8192 --npu force --n 16
echo "=== MTP acceptance (--show path) ==="
sleep 30
timeout 400 ./bin/runz "$M" --n 64 --mtp --mtp-drafts 3 --show "$P" 2>&1 | tail -8
