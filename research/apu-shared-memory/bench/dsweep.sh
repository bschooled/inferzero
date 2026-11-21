#!/usr/bin/env bash
cd ~/inferzero
M=$HOME/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
P="Explain quicksort in one paragraph, then give its complexity."
for N in 3 5 7; do
  echo ===drafts-$N===
  timeout 320 ./bin/runz "$M" --bench-pp 2048 --ctx 2128 --mtp --mtp-drafts $N --n 300 "$P" 2>/dev/null | grep -E "^(pp2048 run 2|tg300|MTP|loaded)"
  sleep 15
done
