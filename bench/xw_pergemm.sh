#!/usr/bin/env bash
# Which XMX tile each of a model's prompt GEMMs wants, measured on this GPU with the model itself (QUEUE: the A770's
# H8 GEMMs, G1). bench/xw_pergemm.ps1 is the same for Windows; the two must stay the same (change both together).
# For every tile setting it runs the real model's prompt (runz --bench-pp, the engine's own kernels on the model's own
# weights) with --prof, and records the GPU time of each GEMM kind (gemm_down, gemm_up, gemm_gate, gemm_qkv, gemm_o,
# gemm_ssm_out, ...): a model's GEMMs differ in shape (down: few columns, a long K; up / gate: many columns), and one
# tile for all of them can hide a tile that wins on some. Every setting gives the same results bit for bit (the engine
# checks the tile at start: "the H8 GEMMs on XMX with ... tiles"; a failed check keeps the default and is reported).
# Two rounds, the settings in the other order the second time; each GEMM's lower time is kept.
# Paste the whole output into the GitHub issue.
#   bash bench/xw_pergemm.sh --model <Qwen3.8 27B or another Qwen dense gguf> [--tokens 4096] [--rounds 2] [--quick]
# Settings: on 8-lane XMX (Arc A770 / A750 / A580, the Arc 140T): the default 32 x 16, 32 x 32, 16 x 32, 16 x 64,
# 64 x 16, and 1 / 4 column sub-groups per work-group; on 16-lane XMX (B-series): the default 32 x 64, 32 x 32,
# 32 x 96, 32 x 128, 16 x 128, 64 x 64, and 1 / 2 column sub-groups. --quick: the first four of each.
# About 1-2 minutes per setting and round for the 27B at 4,096 tokens. Needs bash 4+, awk, grep.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNZ="$ROOT/bin/runz"
die() { echo "$*" >&2; exit 1; }
MODEL=""; TOKENS=4096; ROUNDS=2; QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "${BASH_SOURCE[0]}"; exit 0 ;;
    --model) MODEL="$2"; shift 2 ;;
    --tokens) TOKENS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --quick) QUICK=1; shift ;;
    *) die "unknown option $1 (--help)" ;;
  esac
done
[ -n "$MODEL" ] || die "--model is needed"
[ -f "$MODEL" ] || die "model not found: $MODEL"
[ -x "$RUNZ" ] || die "bin/runz not found: build it first (build.sh)"
VARS=(ARC_GEMM_XW ARC_GEMM_XW_TM ARC_GEMM_XW_NC ARC_GEMM_XW_WGX ARC_GEMM_XW_GRF)
UNSET=(); for v in "${VARS[@]}"; do UNSET+=(-u "$v"); done
TMP="$(mktemp -d)" || die "cannot make a temporary folder"
trap 'rm -rf "$TMP"' EXIT
run() { local e="$1"; shift; env "${UNSET[@]}" $e "$RUNZ" "$MODEL" "$@" > "$TMP/out.txt" 2>&1; }

run "" --bench-pp 1024 --n 0
GPU="$(grep -aE '^inferzero: GPU 0 \(' "$TMP/out.txt" | head -n 1 | sed -E 's/^inferzero: GPU 0 \((.*)\):.*/\1/')"
LANES8=1; grep -aq 'no 8-lane sub-groups' "$TMP/out.txt" && LANES8=0
grep -aqE 'XMX on for the prompt GEMMs|H8 GEMMs on XMX' "$TMP/out.txt" ||
  echo "note: this GPU's log shows no XMX prompt GEMMs; the tile settings below change nothing here"
# name|env
if [ "$LANES8" = 1 ]; then
  SETS=("32x16 (default)|" "32x32|ARC_GEMM_XW_NC=2" "16x32|ARC_GEMM_XW_TM=16 ARC_GEMM_XW_NC=2" "32x16 WGX 4|ARC_GEMM_XW_WGX=4")
  [ "$QUICK" = 1 ] || SETS+=("16x64|ARC_GEMM_XW_TM=16 ARC_GEMM_XW_NC=4" "64x16|ARC_GEMM_XW_TM=64" "32x16 WGX 1|ARC_GEMM_XW_WGX=1"
                             "32x32 WGX 4|ARC_GEMM_XW_NC=2 ARC_GEMM_XW_WGX=4")
  KIND="8-lane XMX (gemm_xw8)"
else
  SETS=("32x64 (default)|" "32x32|ARC_GEMM_XW_NC=2" "32x96|ARC_GEMM_XW_NC=6" "32x128|ARC_GEMM_XW_NC=8")
  [ "$QUICK" = 1 ] || SETS+=("16x128|ARC_GEMM_XW_TM=16 ARC_GEMM_XW_NC=8" "64x64|ARC_GEMM_XW_TM=64" "32x64 WGX 1|ARC_GEMM_XW_WGX=1"
                             "32x64 WGX 2|ARC_GEMM_XW_WGX=2")
  KIND="16-lane XMX (gemm_xw)"
fi
echo "xw_pergemm: $(basename -- "$MODEL"), ${GPU:-GPU ?}, $TOKENS tokens, $ROUNDS round(s), $KIND"

# (an untimed run at the same size first: the first measured setting was ~8 % slower without it)
run "" --bench-pp "$TOKENS" --n 0
# (an untimed run at the same size first: the first measured setting was ~8 % slower without it, on a GPU where every
#  setting runs the same kernels)
run "" --bench-pp "$TOKENS" --n 0
# results: lines "set<TAB>kernel<TAB>ms" and "set<TAB>@pp<TAB>tok/s" and "set<TAB>@note<TAB>text"
RES="$TMP/res.tsv"; : > "$RES"
N=${#SETS[@]}
for ((r = 1; r <= ROUNDS; r++)); do
  for ((i = 0; i < N; i++)); do
    j=$i; [ $((r % 2)) = 0 ] && j=$((N - 1 - i))
    name="${SETS[$j]%%|*}"; e="${SETS[$j]#*|}"
    echo "  round $r: $name ..." >&2
    run "$e" --bench-pp "$TOKENS" --n 0 --prof
    awk -v s="$name" '
      /^[ \t]+gemm_[A-Za-z0-9_]+[ \t]+[0-9.]+ ms/ { printf "%s\t%s\t%s\n", s, $1, $2 }
      /pp[0-9]+ run [0-9]+: [0-9.]+ ms[ \t]+->[ \t]+[0-9.]+ tokens\/s/ { printf "%s\t@pp\t%s\n", s, $(NF-1) }
      /H8 GEMMs on XMX with|gemm_xw.* differs|not supported/ { sub(/^[ \t]+/, ""); printf "%s\t@note\t%s\n", s, $0 }
    ' "$TMP/out.txt" >> "$RES"
  done
done

echo
echo "GPU time per GEMM kind (ms, the lower of the rounds); the last column is each GEMM's fastest setting"
LC_ALL=C awk -F'\t' -v sets="$(printf '%s\n' "${SETS[@]%%|*}")" '
  BEGIN { n = split(sets, S, "\n"); for (i = 1; i <= n; i++) if (S[i] == "") n--; }
  $2 == "@pp" { if (!(($1) in pp) || $3 + 0 > pp[$1]) pp[$1] = $3 + 0; next }
  $2 == "@note" { note[$1] = $3; next }
  { k = $2; if (!(k in seen)) { seen[k] = 1; K[++nk] = k }
    key = $1 SUBSEP k; if (!(key in ms) || $3 + 0 < ms[key]) ms[key] = $3 + 0 }
  END {
    printf "%-16s", "GEMM"; for (i = 1; i <= n; i++) printf " %15s", S[i]; printf "   fastest\n"
    for (q = 1; q <= nk; q++) {
      k = K[q]; printf "%-16s", k; best = ""; bv = 1e30
      for (i = 1; i <= n; i++) { key = S[i] SUBSEP k
        if (key in ms) { printf " %15.1f", ms[key]; if (ms[key] < bv) { bv = ms[key]; best = S[i] } } else printf " %15s", "-" }
      d = S[1] SUBSEP k
      if (d in ms) { sd += ms[d]; sb += bv; printf "   %s (%+.1f %%)\n", best, 100 * (bv / ms[d] - 1) } else printf "   %s\n", best
    }
    printf "%-16s", "prompt tok/s"; for (i = 1; i <= n; i++) printf " %15s", (S[i] in pp) ? sprintf("%.1f", pp[S[i]]) : "failed"
    printf "\n\nAll GEMMs at the default: %.1f ms; each GEMM at its fastest setting: %.1f ms (%+.1f %%).\n", sd, sb, (sd > 0 ? 100 * (sb / sd - 1) : 0)
    print "The tile each setting ran (from the log):"
    for (i = 1; i <= n; i++) printf "  %-16s %s\n", S[i], (S[i] in note) ? note[S[i]] : "(no tile line)"
  }' "$RES"
