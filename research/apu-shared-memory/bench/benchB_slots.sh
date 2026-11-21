#!/usr/bin/env bash
# Experiment B: byte amortization — 2 slots in ONE engine pass on the 40B, MTP on.
cd ~/inferzero
M=~/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
pkill -x serverz 2>/dev/null; sleep 5
./bin/serverz -m "$M" --ctx auto --mtp --slots 2 --host 127.0.0.1 --port 8090 > ~/serverz2.log 2>&1 &
SRV=$!
for i in $(seq 1 90); do curl -s -o /dev/null -m 2 http://127.0.0.1:8090/health && break; sleep 4; done
~/ov/bin/python ~/two_clients.py
sleep 2
grep chatcmpl ~/serverz2.log | tail -4
kill $SRV 2>/dev/null
