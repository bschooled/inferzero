#!/usr/bin/env bash
# Round 7: performance platform profile effect + NPU with render group.
cd ~/inferzero
M=~/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
P="Explain quicksort in one paragraph."
powerprofilesctl set performance
echo "profile_now=$(cat /sys/firmware/acpi/platform_profile)"
sleep 10
echo "=== pp8192 ctx8224 chunk1024 @ performance ==="
timeout 900 ./bin/runz "$M" --bench-pp 8192 --ctx 8224 --chunk 1024 --n 32 "$P" 2>&1 | grep -E "^(loaded|pp[0-9]+|tg[0-9]+|weights|Level Zero)"
sleep 30
echo "=== NPU force @ performance (render group active) ==="
timeout 900 ./bin/runz "$M" --bench-pp 8192 --ctx 8224 --chunk 1024 --npu force --n 32 "$P" 2>&1 | grep -E "^(loaded|pp[0-9]+|tg[0-9]+|weights|Level Zero|ARC_NPU|npu|NPU)"
powerprofilesctl set balanced
echo "profile_restored=$(cat /sys/firmware/acpi/platform_profile)"
