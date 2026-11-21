#!/usr/bin/env bash
# Round 8: real performance-profile comparison + NPU (render group) + server smoke.
cd ~/inferzero
M=~/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-Q4_K_S.gguf
P="Explain quicksort in one paragraph."
echo performance | sudo tee /sys/firmware/acpi/platform_profile >/dev/null
echo "profile_now=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"
sleep 10
echo "=== pp8192 ctx8224 chunk1024 @ PERF profile ==="
timeout 900 ./bin/runz "$M" --bench-pp 8192 --ctx 8224 --chunk 1024 --n 32 "$P" 2>&1 | grep -E "^(pp[0-9]+|tg[0-9]+|weights)"
sleep 30
echo "=== NPU force @ PERF (render group) ==="
timeout 900 ./bin/runz "$M" --bench-pp 8192 --ctx 8224 --chunk 1024 --npu force --n 32 "$P" 2>&1 | grep -E "^(loaded|pp[0-9]+|tg[0-9]+|weights|Level Zero|ARC_NPU|npu|NPU|Segmentation|.*error.*)"; echo "rc=$?"
sleep 30
echo "=== server smoke: serverz --ctx auto --mtp ==="
./bin/serverz -m "$M" --ctx auto --mtp --host 127.0.0.1 --port 8090 > ~/serverz.log 2>&1 &
SRV=$!
for i in $(seq 1 90); do curl -s -o /dev/null -m 2 http://127.0.0.1:8090/health && break; sleep 4; done
echo "health: $(curl -s -m 5 http://127.0.0.1:8090/health)"
LONG=$(python3 -c "print('In software engineering, ' * 120)")
python3 - "$LONG" <<'PY'
import json, sys, time, urllib.request
long = sys.argv[1]
body = {"model": "bench", "messages": [{"role": "user", "content": long + "Summarize the value of unit tests in 6 bullet points."}],
        "max_tokens": 160, "temperature": 0.0}
t0 = time.time()
req = urllib.request.Request("http://127.0.0.1:8090/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
dt = time.time() - t0
u = r["usage"]
print(f"call1: prompt={u['prompt_tokens']} completion={u['completion_tokens']} wall={dt:.1f}s -> {u['completion_tokens']/dt:.2f} tok/s end-to-end")
PY
tail -3 ~/serverz.log
kill $SRV 2>/dev/null
echo balanced | sudo tee /sys/firmware/acpi/platform_profile >/dev/null
echo "profile_restored=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"
