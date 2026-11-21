#!/usr/bin/env bash
# Acceptance checks for goal docs/goals/apu-speedup-qwen38.md.
# usage: bash research/apu-shared-memory/checks/check.sh <criterion-number|all>
# Checks the FORK (origin/main), not the local worktree: work must be merged+pushed
# to count. Conventions (from the goal):
#  - each experiment has research/apu-shared-memory/Pn-*.md with METHOD / NUMBERS /
#    VERDICT sections;
#  - if a VERDICT section says POSITIVE, the file must carry a `Flag: <token>` line
#    and that token must exist in origin/main under zero/ tools/ or bench/;
#  - NEGATIVE / inconclusive verdicts need no code.
set -u
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || { echo "FAIL: not a git repo"; exit 1; }
git fetch -q origin || { echo "FAIL: cannot fetch origin"; exit 1; }
R=origin/main
CRIT=${1:-all}
RC=0
p(){ echo "c$1 OK: $2"; }
f(){ echo "c$1 FAIL: $2"; if [ "$CRIT" != all ]; then exit 1; fi; RC=1; }

tree=$(git ls-tree -r --name-only $R)

verdict(){ # $1 criterion number  $2 file prefix (P0..P5)
  local file body vsec tok
  file=$(printf '%s\n' "$tree" | grep -E "^research/apu-shared-memory/$2-[A-Za-z0-9._-]+\.md$" | head -1)
  if [ -z "$file" ]; then f "$1" "missing research/apu-shared-memory/$2-*.md on origin/main"; return 1; fi
  body=$(git show "$R:$file")
  for s in METHOD NUMBERS VERDICT; do
    case "$body" in *"$s"*) ;; *) f "$1" "$file lacks $s section"; return 1;; esac
  done
  vsec=$(printf '%s\n' "$body" | sed -n '/VERDICT/,$p')
  if printf '%s\n' "$vsec" | grep -qi "POSITIVE"; then
    tok=$(printf '%s\n' "$body" | grep -ioE "flag:? *\.?\`?[A-Za-z0-9_][A-Za-z0-9_.-]*" | head -1 | sed -E 's/[^:]*:[ ]*//; s/`//g; s/^\.//')
    if [ -z "$tok" ]; then f "$1" "$file says POSITIVE but has no 'Flag: <token>' line"; return 1; fi
    if git grep -qi -- "$tok" "$R" -- zero tools bench; then p "$1" "$file + flag '$tok' merged"; return 0; fi
    f "$1" "verdict POSITIVE but flag '$tok' not found in origin/main"; return 1
  fi
  p "$1" "$file (negative/inconclusive verdict recorded; no code required)"; return 0
}

run(){
case "$1" in
 1)
  ok=1
  printf '%s\n' "$tree" | grep -q '^bench/speedbench\.sh$' || { f 1 "bench/speedbench.sh missing"; ok=0; }
  np=$(printf '%s\n' "$tree" | grep -c '^bench/prompts/.*\.txt$')
  [ "${np:-0}" -ge 4 ] || { f 1 "only $np prompt files under bench/prompts/ (need >=4 diverse prose/code/repetition)"; ok=0; }
  b=$(git show "$R:research/apu-shared-memory/BASELINE.md" 2>/dev/null) || { f 1 "research/apu-shared-memory/BASELINE.md missing"; ok=0; }
  for k in IQ2 IQ3 Q4 median; do
    printf '%s' "$b" | grep -qi "$k" || { f 1 "BASELINE.md lacks '$k'"; ok=0; }
  done
  [ $ok = 1 ] && p 1 "diverse-prompt harness + baseline ladder (medians) committed"
  ;;
 2)
  verdict 2 P0 || true
  probe=$(printf '%s\n' "$tree" | grep -i 'busy' | grep -vE '\.md$' | head -3)
  if [ -n "$probe" ]; then p 2 "busy-counter probe code committed: $(printf '%s' "$probe" | tr '\n' ' ')"; else f 2 "no probe code file (any committed path containing 'busy', non-.md)"; fi
  ;;
 3) verdict 3 P1 || true ;;
 4) verdict 4 P2 || true ;;
 5) verdict 5 P3 || true ;;
 6) verdict 6 P4 || true ;;
 7) verdict 7 P5 || true ;;
 8)
  ok=1
  q=$(git show "$R:research/apu-shared-memory/QUALITY-GATES.md" 2>/dev/null) || { f 8 "QUALITY-GATES.md missing"; ok=0; }
  printf '%s' "$q" | grep -qi "self-check" || { f 8 "QUALITY-GATES.md has no W3 startup GEMM self-check result"; ok=0; }
  printf '%s' "$q" | grep -qiE "bit-?exact|greedy" || { f 8 "QUALITY-GATES.md has no greedy-spec==AR bit-exact check"; ok=0; }
  [ $ok = 1 ] && p 8 "quality gates recorded (self-check + greedy bit-exactness)"
  ;;
 9)
  ok=1
  l=$(git show "$R:research/apu-shared-memory/LADDER-FINAL.md" 2>/dev/null) || { f 9 "LADDER-FINAL.md missing"; ok=0; }
  for k in IQ2 IQ3 Q4 "GB/s"; do
    printf '%s' "$l" | grep -q "$k" || { f 9 "LADDER-FINAL.md lacks '$k' regime arithmetic"; ok=0; }
  done
  [ $ok = 1 ] && p 9 "final ladder + per-change regime arithmetic (bytes/pass, GB/s)"
  ;;
esac
}

for c in 1 2 3 4 5 6 7 8 9; do
  if [ "$CRIT" = all ] || [ "$CRIT" = "$c" ]; then run "$c"; fi
done
exit $RC
