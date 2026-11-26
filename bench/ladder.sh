#!/usr/bin/env bash
# The post-merge champion ladder runner used 2026-11-22 (iteration 1, criterion
# 6): four configs back-to-back in one session so the a->b delta is paired.
# Run from an inferzero tree on the node; model paths assume ~/models.
# Results land in research/apu-shared-memory/results/ as speedbench-*-<tag>.log.
set -u
cd "$(dirname "$0")/../../.."   # repo root, wherever this lives
RES="$PWD/research/apu-shared-memory/results"
mkdir -p "$RES"
IQ2="${IQ2:-$HOME/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf}"
IQ3="${IQ3:-$HOME/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ3_M.gguf}"
Q4="${Q4:-$HOME/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf}"
TAG="${TAG:-ladder}"
COMMON=(--full --repeats 3 --cooldown 60 --max-tokens 300 --full-prompts prose,structured --port 8080 --out-dir "$RES")

echo "=== a: IQ2_M d3 (flags off) ==="
SPEEDBENCH_DATE="$TAG-a" bash bench/speedbench.sh "$IQ2" --mtp --mtp-drafts 3 "${COMMON[@]}"
sleep 60
echo "=== b: IQ2_M d3 + --mtp-draft-vocab 32768 ==="
SPEEDBENCH_DATE="$TAG-b" bash bench/speedbench.sh "$IQ2" --mtp --mtp-drafts 3 --mtp-draft-vocab 32768 "${COMMON[@]}"
sleep 60
echo "=== c: IQ3_M d3 (regression) ==="
SPEEDBENCH_DATE="$TAG-c" bash bench/speedbench.sh "$IQ3" --mtp --mtp-drafts 3 "${COMMON[@]}"
sleep 60
echo "=== d: Q4_K_S raw (regression) ==="
SPEEDBENCH_DATE="$TAG-d" bash bench/speedbench.sh "$Q4" "${COMMON[@]}"

if pgrep -x serverz >/dev/null || pgrep -x runz >/dev/null; then echo LEFTOVER_PROC; fi
echo LADDER_DONE
