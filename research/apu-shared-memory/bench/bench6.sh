#!/usr/bin/env bash
# Round 6: 16K context with the working chunk-1024 geometry; power survey midway.
cd ~/inferzero
M=~/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
P="Explain quicksort in one paragraph."
echo "=== --bench-pp 16384 --ctx 16416 --chunk 1024 --n 32 ==="
( sleep 60; bash ~/powsurvey.sh > ~/pow.out 2>&1 ) &
timeout 900 ./bin/runz "$M" --bench-pp 16384 --ctx 16416 --chunk 1024 --n 32 "$P" 2>&1 | grep -E "^(loaded|pp[0-9]+|tg[0-9]+|weights|Level Zero|GPU memory)"
wait
echo "== POWER SURVEY DURING LOAD =="
cat ~/pow.out
