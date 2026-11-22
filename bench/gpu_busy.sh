#!/usr/bin/env bash
# M2-P0: GPU-busy fraction & roofline probe during the IQ2_M+MTP d3 decode window.
# Runs:
#   1. Hardware counter availability probe (gpu_busy_sampler --probe)
#   2. Bit-exactness verification: INFERZERO_GPU_BUSY=0 vs INFERZERO_GPU_BUSY=1 (--dump-logits)
#   3. Screen pass (tg~200, 60s cool before) with INFERZERO_GPU_BUSY=1 + external sampler
#      on representative prompts (prose, structured) using chat-template wrapped prompts
#      to capture per-window CSV, external PMU / fdinfo / RC6 busy %, and per-kernel
#      --prof-gen roofline numbers.
#
# Usage:
#   bash bench/gpu_busy.sh <model.gguf> [--n 200] [--cooldown 60] [--out-dir results]

set -euo pipefail

MODEL="${1:-$HOME/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf}"
shift || true

N_GEN=200
COOLDOWN=60
OUT_DIR="research/apu-shared-memory/results"
PROMPTS=("prose" "structured")

while [ $# -gt 0 ]; do
  case "$1" in
    --n) N_GEN="$2"; shift 2 ;;
    --cooldown) COOLDOWN="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --prompts) IFS=',' read -r -a PROMPTS <<< "$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

[ -f "$MODEL" ] || die "model not found: $MODEL"
[ -x "$ROOT/bin/runz" ] || die "bin/runz not found; run ./build.sh runz serverz gpu_busy_sampler"
[ -x "$ROOT/bin/gpu_busy_sampler" ] || die "bin/gpu_busy_sampler not found; run ./build.sh gpu_busy_sampler"

if pgrep -x serverz >/dev/null 2>&1 || pgrep -x runz >/dev/null 2>&1; then
  die "stray serverz or runz running; aborting"
fi

mkdir -p "$OUT_DIR"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gpu_busy.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

wrap_chat() {
  local src="$1"
  local dst="$2"
  printf '<|im_start|>user\n' > "$dst"
  cat "$src" >> "$dst"
  printf '<|im_end|>\n<|im_start|>assistant\n<think>\n' >> "$dst"
}

echo "=== 1. Hardware Counter Availability Probe ==="
sudo "$ROOT/bin/gpu_busy_sampler" --probe | tee "$OUT_DIR/p0-counter-probe.log"

echo ""
echo "=== 2. Bit-Exactness Check (INFERZERO_GPU_BUSY=0 vs 1, 16 tokens) ==="
wrap_chat "$ROOT/bench/prompts/prose.txt" "$TMP_DIR/prose.chat.txt"
INFERZERO_GPU_BUSY=0 "$ROOT/bin/runz" "$MODEL" --text-file "$TMP_DIR/prose.chat.txt" \
  --ctx 8192 --mtp --mtp-drafts 3 --n 16 --dump-logits "$TMP_DIR/logits_off.bin" \
  > "$TMP_DIR/bitcheck_off.log" 2>&1
INFERZERO_GPU_BUSY=1 INFERZERO_GPU_BUSY_CSV="$TMP_DIR/bitcheck.csv" "$ROOT/bin/runz" "$MODEL" --text-file "$TMP_DIR/prose.chat.txt" \
  --ctx 8192 --mtp --mtp-drafts 3 --n 16 --dump-logits "$TMP_DIR/logits_on.bin" \
  > "$TMP_DIR/bitcheck_on.log" 2>&1

SHA_OFF="$(sha256sum "$TMP_DIR/logits_off.bin" | awk '{print $1}')"
SHA_ON="$(sha256sum "$TMP_DIR/logits_on.bin" | awk '{print $1}')"
echo "sha256(INFERZERO_GPU_BUSY=0): $SHA_OFF"
echo "sha256(INFERZERO_GPU_BUSY=1): $SHA_ON"
if [ "$SHA_OFF" != "$SHA_ON" ]; then
  die "logits differ between INFERZERO_GPU_BUSY=0 and INFERZERO_GPU_BUSY=1"
fi
echo "Bit-identical check PASSED."

for pname in "${PROMPTS[@]}"; do
  pfile="$ROOT/bench/prompts/${pname}.txt"
  [ -f "$pfile" ] || die "prompt file missing: $pfile"
  chat_file="$TMP_DIR/${pname}.chat.txt"
  wrap_chat "$pfile" "$chat_file"

  echo ""
  echo "=== Cooling down ${COOLDOWN}s before ${pname} (tg${N_GEN}) ==="
  sleep "$COOLDOWN"

  if pgrep -x serverz >/dev/null 2>&1 || pgrep -x runz >/dev/null 2>&1; then
    die "stray serverz or runz running before ${pname}; aborting"
  fi

  CSV_OUT="$OUT_DIR/p0-gpu-busy-${pname}.csv"
  LOG_OUT="$OUT_DIR/p0-gpu-busy-${pname}.log"
  EXT_OUT="$OUT_DIR/p0-external-sampler-${pname}.csv"

  echo "=== Running runz --mtp --mtp-drafts 3 --n ${N_GEN} on ${pname} ==="
  INFERZERO_GPU_BUSY=1 INFERZERO_GPU_BUSY_CSV="$CSV_OUT" \
    "$ROOT/bin/runz" "$MODEL" --text-file "$chat_file" \
    --ctx 8192 --mtp --mtp-drafts 3 --n "$N_GEN" --prof-gen --show \
    > "$LOG_OUT" 2>&1 &
  RUNZ_PID=$!

  sudo "$ROOT/bin/gpu_busy_sampler" --pid "$RUNZ_PID" --interval-ms 100 --out "$EXT_OUT" || true
  wait "$RUNZ_PID"

  echo "--- Summary from $LOG_OUT ---"
  grep -E "(prompt:|generation:|weights read:|MTP:|MTP time:|=== INFERZERO_GPU_BUSY|  window total|  draft phase|  verify phase|  verify_done|  pass rate|GPU time by kernel)" "$LOG_OUT" || true
  tail -n 25 "$LOG_OUT" || true
  echo "--- External sampler summary ($EXT_OUT) ---"
  tail -n 3 "$EXT_OUT" || true
done
