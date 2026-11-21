#!/usr/bin/env bash
# Diverse-prompt benchmark client for inferzero (SPEED-Bench style).
# Usage:
#   bash bench/speedbench.sh <model.gguf> [--mtp] [--mtp-drafts N] [--two-phase | --full] [--full-prompts p1,p2] [--repeats N] [--cooldown S] [--max-tokens N] [--screen-repeats N] [--screen-cooldown S] [--screen-tokens N] [--port P] [--out-dir DIR]
#
# Starts one serverz per model (--slots 1 --ctx 8192), sends each prompt from
# bench/prompts/*.txt (excluding *.ids.txt) as a streaming OpenAI chat request
# with temperature=0.0, measures client-side tok/s and captures serverz's own
# t/s and MTP x/y accepted lines, writes raw lines to
# results/speedbench-<config>-<date>.log, and prints a per-prompt + per-config
# median table.
#
# Modes:
#   --two-phase (default):
#     - Full confirmation pass on representative prompts (default: prose,structured):
#       3 repeats, max_tokens=300, 60 s cooldown between runs, 3-repeat medians.
#     - Screen pass on remaining prompts (default: code,fiction):
#       1 repeat, max_tokens=150, 30 s cooldown, showing prompt-to-prompt shape.
#   --full:
#     - Run full confirmation parameters (REPEATS, MAX_TOKENS, COOLDOWN) on every prompt.
#
# Pure bash + curl + awk (no Python).

set -euo pipefail

usage() {
  printf '%s\n' \
    "usage: bash bench/speedbench.sh <model.gguf> [options]" \
    "  <model.gguf>            path to model GGUF" \
    "  --mtp                   enable speculative decoding (MTP)" \
    "  --mtp-drafts N          drafts per step (e.g. 3)" \
    "  --two-phase             full pass on --full-prompts (3x tg300, 60s cool) + screen pass on others (1x tg150, 30s cool) [default]" \
    "  --full                  run full pass parameters on all prompts" \
    "  --full-prompts LIST     comma-separated prompt stems for full pass (default: prose,structured)" \
    "  --repeats N             repeats per full-pass prompt (default: 3)" \
    "  --cooldown S            seconds to sleep before full-pass runs (default: 60)" \
    "  --max-tokens N          generation cap for full-pass runs (default: 300)" \
    "  --screen-repeats N      repeats per screen-pass prompt (default: 1)" \
    "  --screen-cooldown S     seconds to sleep before screen-pass runs (default: 30)" \
    "  --screen-tokens N       generation cap for screen-pass runs (default: 150)" \
    "  --port P                serverz listen port (default: 8090)" \
    "  --out-dir DIR           directory for raw logs (default: results)"
}

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

if [ $# -lt 1 ]; then
  usage >&2
  exit 1
fi

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

MODEL="$1"
shift

MTP=0
MTP_DRAFTS=""
TWO_PHASE=1
FULL_PROMPTS_CSV="prose,structured"
REPEATS=3
COOLDOWN=60
MAX_TOKENS=300
SCREEN_REPEATS=1
SCREEN_COOLDOWN=30
SCREEN_TOKENS=150
PORT=8090
OUT_DIR="results"
EXTRA_SERVER_ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --mtp)
      MTP=1
      shift
      ;;
    --mtp-drafts)
      [ $# -ge 2 ] || die "--mtp-drafts requires a number"
      MTP_DRAFTS="$2"
      shift 2
      ;;
    --two-phase)
      TWO_PHASE=1
      shift
      ;;
    --full)
      TWO_PHASE=0
      shift
      ;;
    --full-prompts)
      [ $# -ge 2 ] || die "--full-prompts requires a comma-separated list"
      FULL_PROMPTS_CSV="$2"
      shift 2
      ;;
    --repeats)
      [ $# -ge 2 ] || die "--repeats requires a number"
      REPEATS="$2"
      shift 2
      ;;
    --cooldown)
      [ $# -ge 2 ] || die "--cooldown requires a number"
      COOLDOWN="$2"
      shift 2
      ;;
    --max-tokens)
      [ $# -ge 2 ] || die "--max-tokens requires a number"
      MAX_TOKENS="$2"
      shift 2
      ;;
    --screen-repeats)
      [ $# -ge 2 ] || die "--screen-repeats requires a number"
      SCREEN_REPEATS="$2"
      shift 2
      ;;
    --screen-cooldown)
      [ $# -ge 2 ] || die "--screen-cooldown requires a number"
      SCREEN_COOLDOWN="$2"
      shift 2
      ;;
    --screen-tokens)
      [ $# -ge 2 ] || die "--screen-tokens requires a number"
      SCREEN_TOKENS="$2"
      shift 2
      ;;
    --port)
      [ $# -ge 2 ] || die "--port requires a number"
      PORT="$2"
      shift 2
      ;;
    --out-dir)
      [ $# -ge 2 ] || die "--out-dir requires a path"
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_SERVER_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -n "${SERVERZ_BIN:-}" ]; then
  SERVERZ="$SERVERZ_BIN"
elif [ -x "$ROOT/bin/serverz" ]; then
  SERVERZ="$ROOT/bin/serverz"
elif [ -x "$HOME/inferzero/bin/serverz" ]; then
  SERVERZ="$HOME/inferzero/bin/serverz"
else
  die "serverz binary not found in $ROOT/bin/serverz or $HOME/inferzero/bin/serverz"
fi

[ -f "$MODEL" ] || die "model file not found: $MODEL"

PROMPT_DIR="${SPEEDBENCH_PROMPTS_DIR:-$SCRIPT_DIR/prompts}"
[ -d "$PROMPT_DIR" ] || die "prompts directory not found: $PROMPT_DIR"

ALL_PROMPTS=()
for pf in "$PROMPT_DIR"/*.txt; do
  [ -e "$pf" ] || continue
  case "$pf" in
    *.ids.txt) continue ;;
  esac
  ALL_PROMPTS+=("$pf")
done
[ ${#ALL_PROMPTS[@]} -ge 4 ] || die "expected >=4 prompt files in $PROMPT_DIR, found ${#ALL_PROMPTS[@]}"

is_full_prompt() {
  local name="$1"
  if [ "$TWO_PHASE" -eq 0 ]; then
    return 0
  fi
  case ",${FULL_PROMPTS_CSV}," in
    *",${name},"*) return 0 ;;
    *) return 1 ;;
  esac
}

# Order prompts so full-confirmation prompts run first, followed by screen prompts
FULL_LIST=()
SCREEN_LIST=()
for pf in "${ALL_PROMPTS[@]}"; do
  pname="$(basename "$pf" .txt)"
  if is_full_prompt "$pname"; then
    FULL_LIST+=("$pf")
  else
    SCREEN_LIST+=("$pf")
  fi
done
PROMPTS=("${FULL_LIST[@]}" "${SCREEN_LIST[@]}")

# Derive short config tag from model filename and MTP flags
MODEL_BASE="$(basename "$MODEL" .gguf)"
QUANT_TAG="$MODEL_BASE"
if [[ "$MODEL_BASE" =~ (IQ[0-9]_[A-Z0-9]+|Q[0-9]_[A-Z0-9_]+|BF16|F16)$ ]]; then
  QUANT_TAG="${BASH_REMATCH[1]}"
fi
if [ "$MTP" -eq 1 ]; then
  if [ -n "$MTP_DRAFTS" ]; then
    CONFIG_TAG="${QUANT_TAG}-mtp-d${MTP_DRAFTS}"
  else
    CONFIG_TAG="${QUANT_TAG}-mtp"
  fi
else
  CONFIG_TAG="${QUANT_TAG}-raw"
fi

DATE_TAG="${SPEEDBENCH_DATE:-$(date +%Y%m%d)}"
mkdir -p "$OUT_DIR"
LOG_FILE="$OUT_DIR/speedbench-${CONFIG_TAG}-${DATE_TAG}.log"

# Non-negotiable measurement discipline: verify no stray serverz or runz is running
if pgrep -x serverz >/dev/null 2>&1 || pgrep -x runz >/dev/null 2>&1; then
  die "stray serverz or runz process detected via pgrep -x; aborting before server start"
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/speedbench.XXXXXX")"
SRV_LOG="$TMP_DIR/serverz.log"
REQ_JSON="$TMP_DIR/req.json"
SSE_OUT="$TMP_DIR/sse.out"
RAW_TSV="$TMP_DIR/raw.tsv"
SRV_PID=""

cleanup() {
  if [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null; then
    kill "$SRV_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SRV_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -9 "$SRV_PID" 2>/dev/null || true
    wait "$SRV_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

# JSON-escape a file's contents into a double-quoted JSON string
json_escape_file() {
  LC_ALL=C awk '
  BEGIN {
    for (i = 1; i < 32; i++) esc[sprintf("%c", i)] = sprintf("\\u%04x", i)
    esc["\""] = "\\\""
    esc["\\"] = "\\\\"
    esc["\n"] = "\\n"
    esc["\r"] = "\\r"
    esc["\t"] = "\\t"
    esc["\b"] = "\\b"
    esc["\f"] = "\\f"
    out = ""
  }
  {
    line = $0
    if (NR > 1) out = out "\\n"
    for (i = 1; i <= length(line); i++) {
      c = substr(line, i, 1)
      out = out ((c in esc) ? esc[c] : c)
    }
  }
  END {
    printf "\"%s\"", out
  }' "$1"
}

SERVER_CMD=("$SERVERZ" -m "$MODEL" --slots 1 --ctx 8192 --host 127.0.0.1 --port "$PORT" --no-prefix-cache --session-cache-gb 0)
if [ "$MTP" -eq 1 ]; then
  SERVER_CMD+=(--mtp)
  if [ -n "$MTP_DRAFTS" ]; then
    SERVER_CMD+=(--mtp-drafts "$MTP_DRAFTS")
  fi
fi
if [ ${#EXTRA_SERVER_ARGS[@]} -gt 0 ]; then
  SERVER_CMD+=("${EXTRA_SERVER_ARGS[@]}")
fi

{
  printf '# speedbench config=%s date=%s\n' "$CONFIG_TAG" "$(date -Iseconds)"
  printf '# model=%s\n' "$MODEL"
  printf '# command=%s\n' "${SERVER_CMD[*]}"
  if [ "$TWO_PHASE" -eq 1 ]; then
    printf '# mode=two-phase full_prompts=%s (repeats=%d cooldown=%ds max_tokens=%d) screen_other (repeats=%d cooldown=%ds max_tokens=%d)\n' \
      "$FULL_PROMPTS_CSV" "$REPEATS" "$COOLDOWN" "$MAX_TOKENS" "$SCREEN_REPEATS" "$SCREEN_COOLDOWN" "$SCREEN_TOKENS"
  else
    printf '# mode=full repeats=%d cooldown=%ds max_tokens=%d\n' "$REPEATS" "$COOLDOWN" "$MAX_TOKENS"
  fi
} > "$LOG_FILE"

SERVER_WORKDIR="$(cd "$(dirname "$SERVERZ")/.." && pwd)"
(
  cd "$SERVER_WORKDIR"
  exec "${SERVER_CMD[@]}"
) > "$SRV_LOG" 2>&1 &
SRV_PID=$!

# Wait for serverz readiness (/health or listening log line)
READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    cat "$SRV_LOG" >> "$LOG_FILE"
    die "serverz exited during startup (see $LOG_FILE)"
  fi
  if grep -q "listening on http://" "$SRV_LOG" 2>/dev/null; then
    if curl -s -o /dev/null -m 2 "http://127.0.0.1:${PORT}/health" 2>/dev/null || \
       curl -s -o /dev/null -m 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null; then
      READY=1
      break
    fi
  fi
  sleep 2
done

if [ "$READY" -ne 1 ]; then
  cat "$SRV_LOG" >> "$LOG_FILE"
  die "serverz did not become ready within timeout"
fi

grep -E "(inferzero built|system:|GPU:|loaded in|model:|serving |listening on)" "$SRV_LOG" >> "$LOG_FILE" || true
printf '\n' >> "$LOG_FILE"

RUN_INDEX=0

for pf in "${PROMPTS[@]}"; do
  pname="$(basename "$pf" .txt)"
  if is_full_prompt "$pname"; then
    P_PHASE="full"
    P_REPEATS="$REPEATS"
    P_COOLDOWN="$COOLDOWN"
    P_MAX_TOKENS="$MAX_TOKENS"
  else
    P_PHASE="screen"
    P_REPEATS="$SCREEN_REPEATS"
    P_COOLDOWN="$SCREEN_COOLDOWN"
    P_MAX_TOKENS="$SCREEN_TOKENS"
  fi

  escaped_prompt="$(json_escape_file "$pf")"
  printf '{"model":"bench","messages":[{"role":"user","content":%s}],"max_tokens":%d,"temperature":0.0,"stream":true,"cache_prompt":false}\n' \
    "$escaped_prompt" "$P_MAX_TOKENS" > "$REQ_JSON"

  for rep in $(seq 1 "$P_REPEATS"); do
    RUN_INDEX=$((RUN_INDEX + 1))
    if [ "$RUN_INDEX" -gt 1 ] && [ "$P_COOLDOWN" -gt 0 ]; then
      sleep "$P_COOLDOWN"
    fi

    prev_lines="$(wc -l < "$SRV_LOG")"
    : > "$SSE_OUT"

    # Measure client-side wall and decode timing on the SSE stream
    TIMING_LINE="$(
      curl -sS -N --no-buffer -H "Content-Type: application/json" \
        --data-binary @"$REQ_JSON" \
        "http://127.0.0.1:${PORT}/v1/chat/completions" \
      | tee "$SSE_OUT" \
      | awk '
        function now_us(   g, line) {
          g = "date +%s%6N"
          g | getline line
          close(g)
          return line + 0
        }
        BEGIN {
          t_start = now_us()
          t_first = 0
          t_last = 0
          chunks = 0
        }
        /^data: \{/ {
          if ($0 ~ /"content":""/ && $0 ~ /"role":"assistant"/) next
          if ($0 ~ /"reasoning_content":/ || $0 ~ /"content":/) {
            t_now = now_us()
            if (t_first == 0) t_first = t_now
            t_last = t_now
            chunks++
          }
        }
        END {
          t_end = now_us()
          if (t_first == 0) { t_first = t_end; t_last = t_end }
          printf "%.4f\t%.4f\t%d\n", (t_end - t_start) / 1e6, (t_last - t_first) / 1e6, chunks
        }'
    )"

    # Wait briefly for serverz to flush its [chatcmpl-arcN] summary line
    SRV_LINE=""
    for _ in $(seq 1 50); do
      SRV_LINE="$(tail -n +"$((prev_lines + 1))" "$SRV_LOG" | grep '\[chatcmpl-arc' | tail -n 1 || true)"
      [ -n "$SRV_LINE" ] && break
      sleep 0.1
    done

    # Parse server log line + final SSE chunk timings
    PARSED="$(
      awk -v srv="$SRV_LINE" -v tline="$TIMING_LINE" '
      BEGIN {
        split(tline, tl, "\t")
        e2e_s = tl[1] + 0
        stream_s = tl[2] + 0
        sse_chunks = tl[3] + 0

        pp_tok = 0; pp_s = 0; pp_tps = 0
        gen_tok = 0; gen_s = 0; srv_tps = 0
        mtp_acc = "-"; mtp_tot = "-"; mtp_xy = "-"

        if (match(srv, /prompt [0-9]+ tok \(cached [0-9]+\) [0-9.]+ s \([0-9.]+ t\/s\)/)) {
          s = substr(srv, RSTART, RLENGTH)
          gsub(/[^0-9. ]/, "", s)
          n = split(s, a, " ")
          pp_tok = a[1] + 0
          pp_s = a[3] + 0
          pp_tps = a[4] + 0
        }
        if (match(srv, /gen [0-9]+ tok [0-9.]+ s \([0-9.]+ t\/s\)/)) {
          s = substr(srv, RSTART, RLENGTH)
          gsub(/[^0-9. ]/, "", s)
          n = split(s, a, " ")
          gen_tok = a[1] + 0
          gen_s = a[2] + 0
          srv_tps = a[3] + 0
        }
        if (match(srv, /MTP [0-9]+\/[0-9]+ drafts accepted/)) {
          s = substr(srv, RSTART, RLENGTH)
          sub(/^MTP /, "", s)
          sub(/ drafts accepted$/, "", s)
          mtp_xy = s
          split(s, m, "/")
          mtp_acc = m[1]
          mtp_tot = m[2]
        }

        client_tps = (stream_s > 0 && gen_tok > 1) ? ((gen_tok - 1) / stream_s) : ((e2e_s > 0) ? (gen_tok / e2e_s) : 0)
        printf "%d\t%.1f\t%d\t%.2f\t%.2f\t%.2f\t%s\n", pp_tok, pp_tps, gen_tok, gen_s, srv_tps, client_tps, mtp_xy
      }'
    )"

    IFS=$'\t' read -r pp_tok pp_tps gen_tok gen_s srv_tps client_tps mtp_xy <<< "$PARSED"

    printf '[%s] phase=%s prompt=%s rep=%d/%d | pp=%s tok (%s t/s) | gen=%s tok in %ss | srv_tps=%s | client_tps=%s | MTP=%s\n' \
      "$CONFIG_TAG" "$P_PHASE" "$pname" "$rep" "$P_REPEATS" "$pp_tok" "$pp_tps" "$gen_tok" "$gen_s" "$srv_tps" "$client_tps" "$mtp_xy" \
      | tee -a "$LOG_FILE"
    if [ -n "$SRV_LINE" ]; then
      printf '  server_log: %s\n' "$SRV_LINE" >> "$LOG_FILE"
    fi

    printf '%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$CONFIG_TAG" "$P_PHASE" "$pname" "$rep" "$pp_tok" "$pp_tps" "$gen_tok" "$gen_s" "$srv_tps" "$client_tps" "$mtp_xy" \
      >> "$RAW_TSV"
  done
done

# Stop serverz and verify it is gone
if [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null; then
  kill "$SRV_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$SRV_PID" 2>/dev/null || break
    sleep 0.5
  done
  kill -9 "$SRV_PID" 2>/dev/null || true
  wait "$SRV_PID" 2>/dev/null || true
fi
SRV_PID=""

if pgrep -x serverz >/dev/null 2>&1; then
  die "serverz still running after shutdown!"
fi

# Compute per-prompt and per-config medians in awk and append summary table to stdout and LOG_FILE
awk -F'\t' '
function sort_num(arr, n,   i, j, tmp) {
  for (i = 2; i <= n; i++) {
    tmp = arr[i] + 0
    j = i - 1
    while (j >= 1 && (arr[j] + 0) > tmp) {
      arr[j + 1] = arr[j]
      j--
    }
    arr[j + 1] = tmp
  }
}
function median(arr, n,   mid) {
  if (n <= 0) return 0
  sort_num(arr, n)
  mid = int((n + 1) / 2)
  if (n % 2 == 1) return arr[mid] + 0
  return ((arr[mid] + 0) + (arr[mid + 1] + 0)) / 2.0
}
function median_idx(vals, labels, n,   i, j, tv, tl, mid) {
  for (i = 2; i <= n; i++) {
    tv = vals[i] + 0
    tl = labels[i]
    j = i - 1
    while (j >= 1 && (vals[j] + 0) > tv) {
      vals[j + 1] = vals[j]
      labels[j + 1] = labels[j]
      j--
    }
    vals[j + 1] = tv
    labels[j + 1] = tl
  }
  mid = int((n + 1) / 2)
  return labels[mid]
}
{
  cfg = $1
  phase = $2
  pr = $3
  rep = $4 + 0
  pp_tok = $5 + 0
  pp_tps = $6 + 0
  gen_tok = $7 + 0
  gen_s = $8 + 0
  srv_tps = $9 + 0
  cli_tps = $10 + 0
  mtp = $11

  if (!(pr in seen_pr)) {
    seen_pr[pr] = 1
    pr_order[++n_pr] = pr
    pr_phase[pr] = phase
  }
  n = ++cnt[pr]
  p_srv[pr, n] = srv_tps
  p_cli[pr, n] = cli_tps
  p_pp[pr, n] = pp_tps
  p_mtp_v[pr, n] = srv_tps
  p_mtp_l[pr, n] = mtp
  p_runs[pr] = (n == 1 ? sprintf("%.2f", srv_tps) : p_runs[pr] ", " sprintf("%.2f", srv_tps))

  if (mtp != "-" && index(mtp, "/") > 0) {
    split(mtp, m, "/")
    tot_acc += m[1] + 0
    tot_drf += m[2] + 0
    p_acc[pr] += m[1] + 0
    p_drf[pr] += m[2] + 0
    if (phase == "full") {
      full_acc += m[1] + 0
      full_drf += m[2] + 0
    }
  }

  if (phase == "full") {
    nf = ++full_n
    full_srv[nf] = srv_tps
    full_cli[nf] = cli_tps
  }
  na = ++all_n
  all_srv[na] = srv_tps
  all_cli[na] = cli_tps
}
END {
  printf "\n=== SPEEDBENCH MEDIAN SUMMARY (%s) ===\n", cfg
  printf "| Config | Phase | Prompt | Runs (srv tok/s) | median srv tok/s | median client tok/s | median pp tok/s | median MTP accepted (total x/y) |\n"
  printf "|---|---|---|---|---:|---:|---:|---|\n"
  n_full_pr = 0
  for (i = 1; i <= n_pr; i++) {
    pr = pr_order[i]
    ph = pr_phase[pr]
    n = cnt[pr]
    delete a_srv; delete a_cli; delete a_pp; delete a_mv; delete a_ml
    for (j = 1; j <= n; j++) {
      a_srv[j] = p_srv[pr, j]
      a_cli[j] = p_cli[pr, j]
      a_pp[j]  = p_pp[pr, j]
      a_mv[j]  = p_mtp_v[pr, j]
      a_ml[j]  = p_mtp_l[pr, j]
    }
    ms = median(a_srv, n)
    mc = median(a_cli, n)
    mp = median(a_pp, n)
    mm = median_idx(a_mv, a_ml, n)
    p_med_srv[i] = ms
    p_med_cli[i] = mc
    p_med_pp[i]  = mp
    if (ph == "full") {
      n_full_pr++
      fp_med_srv[n_full_pr] = ms
      fp_med_cli[n_full_pr] = mc
      fp_med_pp[n_full_pr]  = mp
    }
    if (p_drf[pr] > 0) {
      mtp_str = sprintf("%s (sum %d/%d, %.1f%%)", mm, p_acc[pr], p_drf[pr], 100.0 * p_acc[pr] / p_drf[pr])
    } else {
      mtp_str = "-"
    }
    printf "| %s | %s (%dx) | %s | %s | %.2f | %.2f | %.1f | %s |\n", cfg, ph, n, pr, p_runs[pr], ms, mc, mp, mtp_str
  }
  if (n_full_pr > 0) {
    f_cfg_srv = median(fp_med_srv, n_full_pr)
    f_cfg_cli = median(fp_med_cli, n_full_pr)
    f_cfg_pp  = median(fp_med_pp, n_full_pr)
    f_all_srv = median(full_srv, full_n)
    f_all_cli = median(full_cli, full_n)
    if (full_drf > 0) {
      f_mtp_str = sprintf("sum %d/%d (%.1f%%)", full_acc, full_drf, 100.0 * full_acc / full_drf)
    } else {
      f_mtp_str = "-"
    }
    printf "| **%s** | **FULL-PASS MEDIAN** | **%d full prompts (%d runs)** | — | **%.2f** | **%.2f** | **%.1f** | **%s** |\n", cfg, n_full_pr, full_n, f_all_srv, f_all_cli, f_cfg_pp, f_mtp_str
  }
  all_med_srv = median(all_srv, all_n)
  all_med_cli = median(all_cli, all_n)
  if (tot_drf > 0) {
    tot_mtp_str = sprintf("sum %d/%d (%.1f%%)", tot_acc, tot_drf, 100.0 * tot_acc / tot_drf)
  } else {
    tot_mtp_str = "-"
  }
  printf "| **%s** | **ALL-RUNS MEDIAN** | **%d prompts (%d runs)** | — | **%.2f** | **%.2f** | — | **%s** |\n", cfg, n_pr, all_n, all_med_srv, all_med_cli, tot_mtp_str
}
' "$RAW_TSV" | tee -a "$LOG_FILE"
